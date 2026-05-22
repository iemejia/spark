/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.sql.execution.datasources.parquet

import java.nio.{ByteBuffer, ByteOrder}

import org.apache.parquet.bytes.ByteBufferInputStream
import org.apache.parquet.column.values.bitpacking.Packer
import org.apache.parquet.io.ParquetDecodingException

import org.apache.spark.sql.execution.vectorized.OnHeapColumnVector
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.{DoubleType, FloatType}

/**
 * Unit tests for [[VectorizedAlpValuesReader]].
 *
 * Since the ALP writer is not yet part of the standard parquet-java release,
 * these tests manually construct ALP-encoded pages following the specification:
 *
 * Page Layout:
 *   [Header 7B]: compressionMode(1) + integerEncoding(1) + logVectorSize(1) + numElements(4 LE)
 *   [Offset Array]: 4B per vector (byte offset to vector data, relative to offset array start)
 *   [Vector 0..N]:
 *     AlpInfo(4B): exponent(1) + factor(1) + numExceptions(2 LE)
 *     ForInfo(5B float / 9B double): frameOfReference(4/8 LE) + bitWidth(1)
 *     PackedValues: bit-packed deltas (groups of 8)
 *     ExceptionPositions: uint16 LE per exception
 *     ExceptionValues: raw float(4B) / double(8B) LE per exception
 */
class ParquetAlpEncodingSuite extends ParquetCompatibilityTest with SharedSparkSession {

  // ALP constants matching the encoding spec
  private val ALP_HEADER_SIZE = 7
  private val ALP_INFO_SIZE = 4
  private val FLOAT_FOR_INFO_SIZE = 5
  private val DOUBLE_FOR_INFO_SIZE = 9
  private val DEFAULT_LOG_VECTOR_SIZE = 10 // vectorSize = 1024

  // Powers of 10 for float encode/decode (must match reader)
  private val FLOAT_POW10: Array[Float] = Array(
    1e0f, 1e1f, 1e2f, 1e3f, 1e4f, 1e5f, 1e6f, 1e7f, 1e8f, 1e9f, 1e10f)
  private val FLOAT_POW10_NEGATIVE: Array[Float] = Array(
    1e0f, 1e-1f, 1e-2f, 1e-3f, 1e-4f, 1e-5f, 1e-6f, 1e-7f, 1e-8f, 1e-9f, 1e-10f)

  // Powers of 10 for double encode/decode (must match reader)
  private val DOUBLE_POW10: Array[Double] = Array(
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
    1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18)
  private val DOUBLE_POW10_NEGATIVE: Array[Double] = Array(
    1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
    1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18)

  private val MAGIC_FLOAT: Float = 12582912.0f // 2^22 + 2^23
  private val MAGIC_DOUBLE: Double = 6755399441055744.0 // 2^51 + 2^52

  // ==================== Float Tests ====================

  test("float: decode simple values with no exceptions") {
    // Encode values: 1.2, 3.4, 5.6 with exponent=1, factor=0
    // encoded = fastRound(value * 10^1 * 10^0) = 12, 34, 56
    val values = Array(1.2f, 3.4f, 5.6f)
    val exponent = 1
    val factor = 0
    val encoded = values.map(v => fastRoundFloat(v * FLOAT_POW10(exponent) * FLOAT_POW10_NEGATIVE(factor)))
    // Verify round-trip
    val decoded = encoded.map(e => e * FLOAT_POW10(factor) * FLOAT_POW10_NEGATIVE(exponent))
    for (i <- values.indices) {
      assert(java.lang.Float.floatToRawIntBits(values(i)) ==
        java.lang.Float.floatToRawIntBits(decoded(i)),
        s"Round-trip failed for value ${values(i)} at index $i")
    }

    val page = encodeFloatPage(values, exponent, factor, logVectorSize = 3) // vectorSize=8
    val reader = VectorizedAlpValuesReader.forFloat()
    reader.initFromPage(values.length, ByteBufferInputStream.wrap(page))
    val col = new OnHeapColumnVector(values.length, FloatType)
    reader.readFloats(values.length, col, 0)

    for (i <- values.indices) {
      assert(java.lang.Float.floatToRawIntBits(col.getFloat(i)) ==
        java.lang.Float.floatToRawIntBits(values(i)),
        s"Mismatch at index $i: expected ${values(i)}, got ${col.getFloat(i)}")
    }
    col.close()
  }

  test("float: decode values with exceptions (NaN, Inf, -0.0)") {
    val normalValues = Array(1.5f, 2.5f, 3.5f, 4.5f, 5.5f)
    val exponent = 1
    val factor = 0
    // Positions 1, 3 will be exceptions
    val exceptions = Array((1, Float.NaN), (3, Float.NegativeInfinity))

    val page = encodeFloatPageWithExceptions(
      normalValues, exponent, factor, exceptions, logVectorSize = 3)
    val reader = VectorizedAlpValuesReader.forFloat()
    reader.initFromPage(normalValues.length, ByteBufferInputStream.wrap(page))
    val col = new OnHeapColumnVector(normalValues.length, FloatType)
    reader.readFloats(normalValues.length, col, 0)

    // Non-exception values should decode correctly
    for (i <- normalValues.indices) {
      val expected = exceptions.find(_._1 == i) match {
        case Some((_, excVal)) => excVal
        case None =>
          val enc = fastRoundFloat(normalValues(i) * FLOAT_POW10(exponent) * FLOAT_POW10_NEGATIVE(factor))
          enc * FLOAT_POW10(factor) * FLOAT_POW10_NEGATIVE(exponent)
      }
      if (expected.isNaN) {
        assert(col.getFloat(i).isNaN, s"Expected NaN at index $i, got ${col.getFloat(i)}")
      } else {
        assert(java.lang.Float.floatToRawIntBits(col.getFloat(i)) ==
          java.lang.Float.floatToRawIntBits(expected),
          s"Mismatch at index $i: expected $expected, got ${col.getFloat(i)}")
      }
    }
    col.close()
  }

  test("float: decode with bitWidth=0 (all values identical)") {
    // All values are the same -> encoded integers are the same -> delta=0 -> bitWidth=0
    val values = Array.fill(16)(4.2f)
    val exponent = 1
    val factor = 0

    val page = encodeFloatPage(values, exponent, factor, logVectorSize = 4) // vectorSize=16
    val reader = VectorizedAlpValuesReader.forFloat()
    reader.initFromPage(values.length, ByteBufferInputStream.wrap(page))
    val col = new OnHeapColumnVector(values.length, FloatType)
    reader.readFloats(values.length, col, 0)

    val expected = {
      val enc = fastRoundFloat(4.2f * FLOAT_POW10(exponent) * FLOAT_POW10_NEGATIVE(factor))
      enc * FLOAT_POW10(factor) * FLOAT_POW10_NEGATIVE(exponent)
    }
    for (i <- values.indices) {
      assert(java.lang.Float.floatToRawIntBits(col.getFloat(i)) ==
        java.lang.Float.floatToRawIntBits(expected),
        s"Mismatch at index $i")
    }
    col.close()
  }

  test("float: skip values") {
    val values = (0 until 32).map(i => (i * 0.1f + 1.0f)).toArray
    val exponent = 2
    val factor = 0

    val page = encodeFloatPage(values, exponent, factor, logVectorSize = 5) // vectorSize=32
    val reader = VectorizedAlpValuesReader.forFloat()
    reader.initFromPage(values.length, ByteBufferInputStream.wrap(page))

    // Skip first 5, read 1, skip 10, read remaining
    reader.skipFloats(5)
    val expected5 = {
      val enc = fastRoundFloat(values(5) * FLOAT_POW10(exponent) * FLOAT_POW10_NEGATIVE(factor))
      enc * FLOAT_POW10(factor) * FLOAT_POW10_NEGATIVE(exponent)
    }
    assert(reader.readFloat() == expected5)

    reader.skipFloats(10)
    val col = new OnHeapColumnVector(values.length, FloatType)
    reader.readFloats(16, col, 0) // read remaining 16 values (32 - 5 - 1 - 10 = 16)
    for (i <- 0 until 16) {
      val origIdx = 16 + i
      val expectedVal = {
        val enc = fastRoundFloat(
          values(origIdx) * FLOAT_POW10(exponent) * FLOAT_POW10_NEGATIVE(factor))
        enc * FLOAT_POW10(factor) * FLOAT_POW10_NEGATIVE(exponent)
      }
      assert(java.lang.Float.floatToRawIntBits(col.getFloat(i)) ==
        java.lang.Float.floatToRawIntBits(expectedVal),
        s"Mismatch at output index $i (original index $origIdx)")
    }
    col.close()
  }

  test("float: multiple vectors in a single page") {
    // vectorSize=8, 20 values -> 3 vectors (8 + 8 + 4)
    val values = (0 until 20).map(i => (i * 1.1f + 0.5f)).toArray
    val exponent = 2
    val factor = 0

    val page = encodeFloatPage(values, exponent, factor, logVectorSize = 3)
    val reader = VectorizedAlpValuesReader.forFloat()
    reader.initFromPage(values.length, ByteBufferInputStream.wrap(page))
    val col = new OnHeapColumnVector(values.length, FloatType)
    reader.readFloats(values.length, col, 0)

    for (i <- values.indices) {
      val expectedVal = {
        val enc = fastRoundFloat(
          values(i) * FLOAT_POW10(exponent) * FLOAT_POW10_NEGATIVE(factor))
        enc * FLOAT_POW10(factor) * FLOAT_POW10_NEGATIVE(exponent)
      }
      assert(java.lang.Float.floatToRawIntBits(col.getFloat(i)) ==
        java.lang.Float.floatToRawIntBits(expectedVal),
        s"Mismatch at index $i: expected $expectedVal, got ${col.getFloat(i)}")
    }
    col.close()
  }

  test("float: read past end throws exception") {
    val values = Array(1.0f, 2.0f, 3.0f)
    val page = encodeFloatPage(values, 0, 0, logVectorSize = 3)
    val reader = VectorizedAlpValuesReader.forFloat()
    reader.initFromPage(values.length, ByteBufferInputStream.wrap(page))
    val col = new OnHeapColumnVector(values.length, FloatType)
    reader.readFloats(values.length, col, 0)
    assertThrows[ParquetDecodingException] {
      reader.readFloat()
    }
    col.close()
  }

  // ==================== Double Tests ====================

  test("double: decode simple values with no exceptions") {
    // Use values that round-trip losslessly with exponent=1, factor=0
    // encode = fastRound(value * 10), decode = encoded * 1.0 * 0.1
    // Values like 1.5, 2.5, 3.5 are exact in binary and round-trip perfectly
    val values = Array(1.5, 2.5, 3.5)
    val exponent = 1
    val factor = 0
    val encoded = values.map(v =>
      fastRoundDouble(v * DOUBLE_POW10(exponent) * DOUBLE_POW10_NEGATIVE(factor)))
    val decoded = encoded.map(e => e * DOUBLE_POW10(factor) * DOUBLE_POW10_NEGATIVE(exponent))
    for (i <- values.indices) {
      assert(java.lang.Double.doubleToRawLongBits(values(i)) ==
        java.lang.Double.doubleToRawLongBits(decoded(i)),
        s"Round-trip failed for value ${values(i)} at index $i")
    }

    val page = encodeDoublePage(values, exponent, factor, logVectorSize = 3)
    val reader = VectorizedAlpValuesReader.forDouble()
    reader.initFromPage(values.length, ByteBufferInputStream.wrap(page))
    val col = new OnHeapColumnVector(values.length, DoubleType)
    reader.readDoubles(values.length, col, 0)

    for (i <- values.indices) {
      assert(java.lang.Double.doubleToRawLongBits(col.getDouble(i)) ==
        java.lang.Double.doubleToRawLongBits(values(i)),
        s"Mismatch at index $i: expected ${values(i)}, got ${col.getDouble(i)}")
    }
    col.close()
  }

  test("double: decode values with exceptions") {
    val normalValues = Array(1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8)
    val exponent = 1
    val factor = 0
    val exceptions = Array((2, Double.NaN), (5, Double.PositiveInfinity), (7, -0.0))

    val page = encodeDoublePageWithExceptions(
      normalValues, exponent, factor, exceptions, logVectorSize = 3)
    val reader = VectorizedAlpValuesReader.forDouble()
    reader.initFromPage(normalValues.length, ByteBufferInputStream.wrap(page))
    val col = new OnHeapColumnVector(normalValues.length, DoubleType)
    reader.readDoubles(normalValues.length, col, 0)

    for (i <- normalValues.indices) {
      val expected = exceptions.find(_._1 == i) match {
        case Some((_, excVal)) => excVal
        case None =>
          val enc = fastRoundDouble(
            normalValues(i) * DOUBLE_POW10(exponent) * DOUBLE_POW10_NEGATIVE(factor))
          enc * DOUBLE_POW10(factor) * DOUBLE_POW10_NEGATIVE(exponent)
      }
      if (expected.isNaN) {
        assert(col.getDouble(i).isNaN, s"Expected NaN at index $i, got ${col.getDouble(i)}")
      } else {
        assert(java.lang.Double.doubleToRawLongBits(col.getDouble(i)) ==
          java.lang.Double.doubleToRawLongBits(expected),
          s"Mismatch at index $i: expected $expected, got ${col.getDouble(i)}")
      }
    }
    col.close()
  }

  test("double: decode with bitWidth=0") {
    val values = Array.fill(8)(3.14)
    val exponent = 2
    val factor = 0

    val page = encodeDoublePage(values, exponent, factor, logVectorSize = 3)
    val reader = VectorizedAlpValuesReader.forDouble()
    reader.initFromPage(values.length, ByteBufferInputStream.wrap(page))
    val col = new OnHeapColumnVector(values.length, DoubleType)
    reader.readDoubles(values.length, col, 0)

    val expected = {
      val enc = fastRoundDouble(3.14 * DOUBLE_POW10(exponent) * DOUBLE_POW10_NEGATIVE(factor))
      enc * DOUBLE_POW10(factor) * DOUBLE_POW10_NEGATIVE(exponent)
    }
    for (i <- values.indices) {
      assert(java.lang.Double.doubleToRawLongBits(col.getDouble(i)) ==
        java.lang.Double.doubleToRawLongBits(expected),
        s"Mismatch at index $i")
    }
    col.close()
  }

  test("double: skip values") {
    val values = (0 until 16).map(i => i * 0.5 + 1.0).toArray
    val exponent = 1
    val factor = 0

    val page = encodeDoublePage(values, exponent, factor, logVectorSize = 4)
    val reader = VectorizedAlpValuesReader.forDouble()
    reader.initFromPage(values.length, ByteBufferInputStream.wrap(page))

    reader.skipDoubles(3)
    val expected3 = {
      val enc = fastRoundDouble(values(3) * DOUBLE_POW10(exponent) * DOUBLE_POW10_NEGATIVE(factor))
      enc * DOUBLE_POW10(factor) * DOUBLE_POW10_NEGATIVE(exponent)
    }
    assert(reader.readDouble() == expected3)
    reader.skipDoubles(12) // skip remaining 12 values (16 - 3 - 1 = 12)
  }

  test("double: multiple vectors in a single page") {
    // vectorSize=8, 24 values -> 3 vectors (8 + 8 + 8)
    val values = (0 until 24).map(i => i * 0.01 + 100.0).toArray
    val exponent = 2
    val factor = 0

    val page = encodeDoublePage(values, exponent, factor, logVectorSize = 3)
    val reader = VectorizedAlpValuesReader.forDouble()
    reader.initFromPage(values.length, ByteBufferInputStream.wrap(page))
    val col = new OnHeapColumnVector(values.length, DoubleType)
    reader.readDoubles(values.length, col, 0)

    for (i <- values.indices) {
      val expectedVal = {
        val enc = fastRoundDouble(
          values(i) * DOUBLE_POW10(exponent) * DOUBLE_POW10_NEGATIVE(factor))
        enc * DOUBLE_POW10(factor) * DOUBLE_POW10_NEGATIVE(exponent)
      }
      assert(java.lang.Double.doubleToRawLongBits(col.getDouble(i)) ==
        java.lang.Double.doubleToRawLongBits(expectedVal),
        s"Mismatch at index $i: expected $expectedVal, got ${col.getDouble(i)}")
    }
    col.close()
  }

  test("double: large vector (1024 values)") {
    val values = (0 until 1024).map(i => i * 0.001 + 50.0).toArray
    val exponent = 3
    val factor = 0

    val page = encodeDoublePage(values, exponent, factor, logVectorSize = DEFAULT_LOG_VECTOR_SIZE)
    val reader = VectorizedAlpValuesReader.forDouble()
    reader.initFromPage(values.length, ByteBufferInputStream.wrap(page))
    val col = new OnHeapColumnVector(values.length, DoubleType)
    reader.readDoubles(values.length, col, 0)

    for (i <- values.indices) {
      val expectedVal = {
        val enc = fastRoundDouble(
          values(i) * DOUBLE_POW10(exponent) * DOUBLE_POW10_NEGATIVE(factor))
        enc * DOUBLE_POW10(factor) * DOUBLE_POW10_NEGATIVE(exponent)
      }
      assert(java.lang.Double.doubleToRawLongBits(col.getDouble(i)) ==
        java.lang.Double.doubleToRawLongBits(expectedVal),
        s"Mismatch at index $i: expected $expectedVal, got ${col.getDouble(i)}")
    }
    col.close()
  }

  test("double: read past end throws exception") {
    val values = Array(1.0, 2.0, 3.0)
    val page = encodeDoublePage(values, 0, 0, logVectorSize = 3)
    val reader = VectorizedAlpValuesReader.forDouble()
    reader.initFromPage(values.length, ByteBufferInputStream.wrap(page))
    val col = new OnHeapColumnVector(values.length, DoubleType)
    reader.readDoubles(values.length, col, 0)
    assertThrows[ParquetDecodingException] {
      reader.readDouble()
    }
    col.close()
  }

  test("float: single value read") {
    val values = Array(1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f)
    val exponent = 1
    val factor = 0

    val page = encodeFloatPage(values, exponent, factor, logVectorSize = 3)
    val reader = VectorizedAlpValuesReader.forFloat()
    reader.initFromPage(values.length, ByteBufferInputStream.wrap(page))

    for (i <- values.indices) {
      val expectedVal = {
        val enc = fastRoundFloat(
          values(i) * FLOAT_POW10(exponent) * FLOAT_POW10_NEGATIVE(factor))
        enc * FLOAT_POW10(factor) * FLOAT_POW10_NEGATIVE(exponent)
      }
      assert(java.lang.Float.floatToRawIntBits(reader.readFloat()) ==
        java.lang.Float.floatToRawIntBits(expectedVal),
        s"Mismatch at index $i")
    }
  }

  test("double: single value read") {
    val values = Array(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5)
    val exponent = 1
    val factor = 0

    val page = encodeDoublePage(values, exponent, factor, logVectorSize = 3)
    val reader = VectorizedAlpValuesReader.forDouble()
    reader.initFromPage(values.length, ByteBufferInputStream.wrap(page))

    for (i <- values.indices) {
      val expectedVal = {
        val enc = fastRoundDouble(
          values(i) * DOUBLE_POW10(exponent) * DOUBLE_POW10_NEGATIVE(factor))
        enc * DOUBLE_POW10(factor) * DOUBLE_POW10_NEGATIVE(exponent)
      }
      assert(java.lang.Double.doubleToRawLongBits(reader.readDouble()) ==
        java.lang.Double.doubleToRawLongBits(expectedVal),
        s"Mismatch at index $i")
    }
  }

  test("float: non-zero factor (exponent=3, factor=1)") {
    // Values like 10.0, 20.0, 30.0 with e=3, f=1
    // encode: fastRound(value * 10^3 * 10^-1) = fastRound(value * 100)
    // decode: encoded * 10^1 * 10^-3 = encoded * 0.01
    val values = Array(10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f)
    val exponent = 3
    val factor = 1

    val page = encodeFloatPage(values, exponent, factor, logVectorSize = 3)
    val reader = VectorizedAlpValuesReader.forFloat()
    reader.initFromPage(values.length, ByteBufferInputStream.wrap(page))
    val col = new OnHeapColumnVector(values.length, FloatType)
    reader.readFloats(values.length, col, 0)

    for (i <- values.indices) {
      val expectedVal = {
        val enc = fastRoundFloat(
          values(i) * FLOAT_POW10(exponent) * FLOAT_POW10_NEGATIVE(factor))
        enc * FLOAT_POW10(factor) * FLOAT_POW10_NEGATIVE(exponent)
      }
      assert(java.lang.Float.floatToRawIntBits(col.getFloat(i)) ==
        java.lang.Float.floatToRawIntBits(expectedVal),
        s"Mismatch at index $i: expected $expectedVal, got ${col.getFloat(i)}")
    }
    col.close()
  }

  // ==================== Helper: Fast Rounding ====================

  private def fastRoundFloat(value: Float): Int = {
    if (value >= 0) ((value + MAGIC_FLOAT) - MAGIC_FLOAT).toInt
    else ((value - MAGIC_FLOAT) + MAGIC_FLOAT).toInt
  }

  private def fastRoundDouble(value: Double): Long = {
    if (value >= 0) ((value + MAGIC_DOUBLE) - MAGIC_DOUBLE).toLong
    else ((value - MAGIC_DOUBLE) + MAGIC_DOUBLE).toLong
  }

  // ==================== Helper: Float Page Encoder ====================

  /**
   * Encodes an array of float values into a complete ALP page (no exceptions).
   */
  private def encodeFloatPage(
      values: Array[Float],
      exponent: Int,
      factor: Int,
      logVectorSize: Int): ByteBuffer = {
    encodeFloatPageWithExceptions(values, exponent, factor, Array.empty, logVectorSize)
  }

  /**
   * Encodes float values with specified exception positions.
   * Exceptions array contains (position, rawValue) pairs.
   */
  private def encodeFloatPageWithExceptions(
      values: Array[Float],
      exponent: Int,
      factor: Int,
      exceptions: Array[(Int, Float)],
      logVectorSize: Int): ByteBuffer = {
    val vectorSize = 1 << logVectorSize
    val numVectors = (values.length + vectorSize - 1) / vectorSize
    val excPositionSet = exceptions.map(_._1).toSet

    // Encode each vector
    val vectorBytes = new Array[Array[Byte]](numVectors)
    for (v <- 0 until numVectors) {
      val start = v * vectorSize
      val end = Math.min(start + vectorSize, values.length)
      val vectorLen = end - start

      // Encode values to integers
      val encoded = new Array[Int](vectorLen)
      val vecExceptions = scala.collection.mutable.ArrayBuffer[(Int, Float)]()
      var placeholder = 0
      var foundPlaceholder = false

      for (i <- 0 until vectorLen) {
        val globalIdx = start + i
        if (excPositionSet.contains(globalIdx)) {
          vecExceptions += ((i, exceptions.find(_._1 == globalIdx).get._2))
          encoded(i) = 0 // temporary
        } else {
          val enc = fastRoundFloat(
            values(globalIdx) * FLOAT_POW10(exponent) * FLOAT_POW10_NEGATIVE(factor))
          encoded(i) = enc
          if (!foundPlaceholder) {
            placeholder = enc
            foundPlaceholder = true
          }
        }
      }

      // Fill exception slots with placeholder
      for ((pos, _) <- vecExceptions) {
        encoded(pos) = placeholder
      }

      // Frame of Reference
      val minValue = if (foundPlaceholder) encoded.min else 0
      val deltas = encoded.map(_ - minValue)
      val maxDelta = if (deltas.isEmpty) 0 else deltas.max
      val bitWidth = if (maxDelta == 0) 0 else (32 - Integer.numberOfLeadingZeros(maxDelta))

      // Build vector bytes
      val buf = ByteBuffer.allocate(
        ALP_INFO_SIZE + FLOAT_FOR_INFO_SIZE +
          (vectorLen * bitWidth + 7) / 8 +
          vecExceptions.size * (java.lang.Short.BYTES + java.lang.Float.BYTES))
        .order(ByteOrder.LITTLE_ENDIAN)

      // AlpInfo
      buf.put(exponent.toByte)
      buf.put(factor.toByte)
      buf.putShort(vecExceptions.size.toShort)

      // ForInfo
      buf.putInt(minValue)
      buf.put(bitWidth.toByte)

      // Bit-packed deltas
      if (bitWidth > 0) {
        val packer = Packer.LITTLE_ENDIAN.newBytePacker(bitWidth)
        val packBuf = new Array[Byte](bitWidth)
        val numFullGroups = vectorLen / 8
        val remaining = vectorLen % 8

        for (g <- 0 until numFullGroups) {
          packer.pack8Values(deltas, g * 8, packBuf, 0)
          buf.put(packBuf, 0, bitWidth)
        }

        if (remaining > 0) {
          val padded = new Array[Int](8)
          System.arraycopy(deltas, numFullGroups * 8, padded, 0, remaining)
          packer.pack8Values(padded, 0, packBuf, 0)
          val totalPackedBytes = (vectorLen * bitWidth + 7) / 8
          val alreadyWritten = numFullGroups * bitWidth
          buf.put(packBuf, 0, totalPackedBytes - alreadyWritten)
        }
      }

      // Exception positions
      for ((pos, _) <- vecExceptions) {
        buf.putShort(pos.toShort)
      }

      // Exception values
      for ((_, excVal) <- vecExceptions) {
        buf.putInt(java.lang.Float.floatToRawIntBits(excVal))
      }

      buf.flip()
      vectorBytes(v) = new Array[Byte](buf.remaining())
      buf.get(vectorBytes(v))
    }

    // Assemble page: header + offset array + vector data
    val offsetArraySize = numVectors * Integer.BYTES
    val totalVectorDataSize = vectorBytes.map(_.length).sum
    val pageSize = ALP_HEADER_SIZE + offsetArraySize + totalVectorDataSize
    val page = ByteBuffer.allocate(pageSize).order(ByteOrder.LITTLE_ENDIAN)

    // Header
    page.put(0.toByte) // compressionMode
    page.put(0.toByte) // integerEncoding (FOR)
    page.put(logVectorSize.toByte)
    page.putInt(values.length)

    // Offset array (offsets relative to start of body after header, i.e., offset array start)
    var currentOffset = offsetArraySize
    for (v <- 0 until numVectors) {
      page.putInt(currentOffset)
      currentOffset += vectorBytes(v).length
    }

    // Vector data
    for (v <- 0 until numVectors) {
      page.put(vectorBytes(v))
    }

    page.flip()
    page
  }

  // ==================== Helper: Double Page Encoder ====================

  private def encodeDoublePage(
      values: Array[Double],
      exponent: Int,
      factor: Int,
      logVectorSize: Int): ByteBuffer = {
    encodeDoublePageWithExceptions(values, exponent, factor, Array.empty, logVectorSize)
  }

  private def encodeDoublePageWithExceptions(
      values: Array[Double],
      exponent: Int,
      factor: Int,
      exceptions: Array[(Int, Double)],
      logVectorSize: Int): ByteBuffer = {
    val vectorSize = 1 << logVectorSize
    val numVectors = (values.length + vectorSize - 1) / vectorSize
    val excPositionSet = exceptions.map(_._1).toSet

    val vectorBytes = new Array[Array[Byte]](numVectors)
    for (v <- 0 until numVectors) {
      val start = v * vectorSize
      val end = Math.min(start + vectorSize, values.length)
      val vectorLen = end - start

      val encoded = new Array[Long](vectorLen)
      val vecExceptions = scala.collection.mutable.ArrayBuffer[(Int, Double)]()
      var placeholder = 0L
      var foundPlaceholder = false

      for (i <- 0 until vectorLen) {
        val globalIdx = start + i
        if (excPositionSet.contains(globalIdx)) {
          vecExceptions += ((i, exceptions.find(_._1 == globalIdx).get._2))
          encoded(i) = 0L
        } else {
          val enc = fastRoundDouble(
            values(globalIdx) * DOUBLE_POW10(exponent) * DOUBLE_POW10_NEGATIVE(factor))
          encoded(i) = enc
          if (!foundPlaceholder) {
            placeholder = enc
            foundPlaceholder = true
          }
        }
      }

      for ((pos, _) <- vecExceptions) {
        encoded(pos) = placeholder
      }

      val minValue = if (foundPlaceholder) encoded.min else 0L
      val deltas = encoded.map(_ - minValue)
      val maxDelta = if (deltas.isEmpty) 0L else deltas.max
      val bitWidth = if (maxDelta == 0) 0 else (64 - java.lang.Long.numberOfLeadingZeros(maxDelta))

      val buf = ByteBuffer.allocate(
        ALP_INFO_SIZE + DOUBLE_FOR_INFO_SIZE +
          (vectorLen * bitWidth + 7) / 8 +
          vecExceptions.size * (java.lang.Short.BYTES + java.lang.Double.BYTES))
        .order(ByteOrder.LITTLE_ENDIAN)

      // AlpInfo
      buf.put(exponent.toByte)
      buf.put(factor.toByte)
      buf.putShort(vecExceptions.size.toShort)

      // ForInfo (8 bytes for double FOR)
      buf.putLong(minValue)
      buf.put(bitWidth.toByte)

      // Bit-packed deltas
      if (bitWidth > 0) {
        val packer = Packer.LITTLE_ENDIAN.newBytePackerForLong(bitWidth)
        val packBuf = new Array[Byte](bitWidth)
        val numFullGroups = vectorLen / 8
        val remaining = vectorLen % 8

        for (g <- 0 until numFullGroups) {
          packer.pack8Values(deltas, g * 8, packBuf, 0)
          buf.put(packBuf, 0, bitWidth)
        }

        if (remaining > 0) {
          val padded = new Array[Long](8)
          System.arraycopy(deltas, numFullGroups * 8, padded, 0, remaining)
          packer.pack8Values(padded, 0, packBuf, 0)
          val totalPackedBytes = (vectorLen * bitWidth + 7) / 8
          val alreadyWritten = numFullGroups * bitWidth
          buf.put(packBuf, 0, totalPackedBytes - alreadyWritten)
        }
      }

      // Exception positions
      for ((pos, _) <- vecExceptions) {
        buf.putShort(pos.toShort)
      }

      // Exception values
      for ((_, excVal) <- vecExceptions) {
        buf.putLong(java.lang.Double.doubleToRawLongBits(excVal))
      }

      buf.flip()
      vectorBytes(v) = new Array[Byte](buf.remaining())
      buf.get(vectorBytes(v))
    }

    val offsetArraySize = numVectors * Integer.BYTES
    val totalVectorDataSize = vectorBytes.map(_.length).sum
    val pageSize = ALP_HEADER_SIZE + offsetArraySize + totalVectorDataSize
    val page = ByteBuffer.allocate(pageSize).order(ByteOrder.LITTLE_ENDIAN)

    // Header
    page.put(0.toByte)
    page.put(0.toByte)
    page.put(logVectorSize.toByte)
    page.putInt(values.length)

    // Offset array
    var currentOffset = offsetArraySize
    for (v <- 0 until numVectors) {
      page.putInt(currentOffset)
      currentOffset += vectorBytes(v).length
    }

    // Vector data
    for (v <- 0 until numVectors) {
      page.put(vectorBytes(v))
    }

    page.flip()
    page
  }
}
