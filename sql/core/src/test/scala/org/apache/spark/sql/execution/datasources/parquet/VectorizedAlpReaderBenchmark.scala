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

import java.nio.ByteBuffer

import scala.util.Random

import org.apache.parquet.bytes.{ByteBufferInputStream, DirectByteBufferAllocator}
import org.apache.parquet.column.values.alp.AlpValuesWriter

import org.apache.spark.benchmark.{Benchmark, BenchmarkBase}
import org.apache.spark.sql.execution.vectorized.OnHeapColumnVector
import org.apache.spark.sql.types.{DoubleType, FloatType}

/**
 * Low-level benchmark for the ALP (Adaptive Lossless floating-Point) Parquet decoder:
 * `VectorizedAlpValuesReader`.
 *
 * Measures decode throughput across five data distributions representing the ALP
 * performance spectrum from best-case (INTEGER) to worst-case (HIGH_EXCEPTION):
 *
 *   - INTEGER: whole numbers - zero exceptions, narrow bit-width.
 *   - MONETARY: 2 decimal places - ideal ALP target, zero exceptions.
 *   - SENSOR: 4-6 significant digits with varying exponents - broader encoding params.
 *   - RANDOM: uniform random floating-point - wide bit-width, stresses bit-packing.
 *   - HIGH_EXCEPTION: ~30% exception rate - subnormals, NaN, Inf mixed with normals.
 *
 * Groups:
 *   A. ALP DOUBLE decode - readDoubles / skipDoubles across distributions.
 *   B. ALP FLOAT decode - readFloats / skipFloats across distributions.
 *   C. Single-value reads - per-call overhead of readFloat / readDouble.
 *
 * To run this benchmark:
 * {{{
 *   1. build/sbt "sql/Test/runMain <this class>"
 *   2. generate result:
 *      SPARK_GENERATE_BENCHMARK_FILES=1 build/sbt "sql/Test/runMain <this class>"
 *      Results in "benchmarks/VectorizedAlpReaderBenchmark-results.txt".
 *   3. GHA: `Run benchmarks` workflow, class = `*VectorizedAlpReader*`.
 * }}}
 */
object VectorizedAlpReaderBenchmark extends BenchmarkBase {

  private val NUM_ROWS = 1024 * 1024
  private val NUM_ITERS = 5
  private val INITIAL_CAPACITY = 4 * 1024 * 1024
  private val PAGE_SIZE = 4 * 1024 * 1024

  // --------------- Data Generation ---------------

  private def generateFloatData(distribution: String, numValues: Int, rng: Random): Array[Float] = {
    distribution match {
      case "integer" =>
        Array.tabulate(numValues)(_ => rng.nextInt(100000).toFloat)
      case "monetary" =>
        Array.tabulate(numValues)(_ => Math.round(rng.nextFloat() * 999999) / 100.0f)
      case "sensor" =>
        Array.tabulate(numValues) { _ =>
          val scale = Math.pow(10, rng.nextInt(8) - 3).toFloat
          Math.round(rng.nextFloat() * 100000) / 10.0f * scale
        }
      case "random" =>
        Array.tabulate(numValues)(_ => rng.nextFloat() * 100000.0f - 50000.0f)
      case "high-exception" =>
        Array.tabulate(numValues) { _ =>
          if (rng.nextInt(10) < 3) {
            // ~30% exceptions
            rng.nextInt(5) match {
              case 0 => Float.NaN
              case 1 => java.lang.Float.intBitsToFloat(rng.nextInt(0x007FFFFF) + 1) // subnormal
              case 2 => if (rng.nextBoolean()) Float.PositiveInfinity else Float.NegativeInfinity
              case 3 => -0.0f
              case 4 => java.lang.Float.intBitsToFloat(rng.nextInt()) // random bits
            }
          } else {
            Math.round(rng.nextFloat() * 99999) / 100.0f
          }
        }
    }
  }

  private def generateDoubleData(
      distribution: String, numValues: Int, rng: Random): Array[Double] = {
    distribution match {
      case "integer" =>
        Array.tabulate(numValues)(_ => rng.nextInt(100000).toDouble)
      case "monetary" =>
        Array.tabulate(numValues)(_ => Math.round(rng.nextDouble() * 999999) / 100.0)
      case "sensor" =>
        Array.tabulate(numValues) { _ =>
          val scale = Math.pow(10, rng.nextInt(8) - 3)
          Math.round(rng.nextDouble() * 1000000) / 100.0 * scale
        }
      case "random" =>
        Array.tabulate(numValues)(_ => rng.nextDouble() * 100000.0 - 50000.0)
      case "high-exception" =>
        Array.tabulate(numValues) { _ =>
          if (rng.nextInt(10) < 3) {
            rng.nextInt(5) match {
              case 0 => Double.NaN
              case 1 =>
                java.lang.Double.longBitsToDouble(rng.nextLong() & 0x000FFFFFFFFFFFFFL) // subnormal
              case 2 =>
                if (rng.nextBoolean()) Double.PositiveInfinity else Double.NegativeInfinity
              case 3 => -0.0
              case 4 => java.lang.Double.longBitsToDouble(rng.nextLong()) // random bits
            }
          } else {
            Math.round(rng.nextDouble() * 99999) / 100.0
          }
        }
    }
  }

  // --------------- ALP Encoding via parquet-java writer ---------------

  private def encodeAlpFloats(values: Array[Float]): Array[Byte] = {
    val writer = new AlpValuesWriter.FloatAlpValuesWriter(
      INITIAL_CAPACITY, PAGE_SIZE, new DirectByteBufferAllocator)
    try {
      var i = 0
      while (i < values.length) { writer.writeFloat(values(i)); i += 1 }
      writer.getBytes.toByteArray
    } finally {
      writer.close()
    }
  }

  private def encodeAlpDoubles(values: Array[Double]): Array[Byte] = {
    val writer = new AlpValuesWriter.DoubleAlpValuesWriter(
      INITIAL_CAPACITY, PAGE_SIZE, new DirectByteBufferAllocator)
    try {
      var i = 0
      while (i < values.length) { writer.writeDouble(values(i)); i += 1 }
      writer.getBytes.toByteArray
    } finally {
      writer.close()
    }
  }

  // --------------- Group A: ALP DOUBLE decode ---------------

  private def runAlpDoubleBenchmark(): Unit = {
    val benchmark = new Benchmark(
      "ALP DOUBLE decode", NUM_ROWS.toLong, NUM_ITERS, output = output)
    val vec = new OnHeapColumnVector(NUM_ROWS, DoubleType)

    val rng = new Random(42)
    val distributions = Seq("integer", "monetary", "sensor", "random", "high-exception")

    distributions.foreach { tag =>
      val values = generateDoubleData(tag, NUM_ROWS, rng)
      val bytes = encodeAlpDoubles(values)

      // Pre-warm
      val warm = VectorizedAlpValuesReader.forDouble()
      warm.initFromPage(NUM_ROWS, ByteBufferInputStream.wrap(ByteBuffer.wrap(bytes)))
      warm.readDoubles(NUM_ROWS, vec, 0)

      benchmark.addCase(s"readDoubles, $tag") { _ =>
        val r = VectorizedAlpValuesReader.forDouble()
        r.initFromPage(NUM_ROWS, ByteBufferInputStream.wrap(ByteBuffer.wrap(bytes)))
        r.readDoubles(NUM_ROWS, vec, 0)
      }

      benchmark.addCase(s"skipDoubles, $tag") { _ =>
        val r = VectorizedAlpValuesReader.forDouble()
        r.initFromPage(NUM_ROWS, ByteBufferInputStream.wrap(ByteBuffer.wrap(bytes)))
        r.skipDoubles(NUM_ROWS)
      }
    }
    benchmark.run()
  }

  // --------------- Group B: ALP FLOAT decode ---------------

  private def runAlpFloatBenchmark(): Unit = {
    val benchmark = new Benchmark(
      "ALP FLOAT decode", NUM_ROWS.toLong, NUM_ITERS, output = output)
    val vec = new OnHeapColumnVector(NUM_ROWS, FloatType)

    val rng = new Random(42)
    val distributions = Seq("integer", "monetary", "sensor", "random", "high-exception")

    distributions.foreach { tag =>
      val values = generateFloatData(tag, NUM_ROWS, rng)
      val bytes = encodeAlpFloats(values)

      // Pre-warm
      val warm = VectorizedAlpValuesReader.forFloat()
      warm.initFromPage(NUM_ROWS, ByteBufferInputStream.wrap(ByteBuffer.wrap(bytes)))
      warm.readFloats(NUM_ROWS, vec, 0)

      benchmark.addCase(s"readFloats, $tag") { _ =>
        val r = VectorizedAlpValuesReader.forFloat()
        r.initFromPage(NUM_ROWS, ByteBufferInputStream.wrap(ByteBuffer.wrap(bytes)))
        r.readFloats(NUM_ROWS, vec, 0)
      }

      benchmark.addCase(s"skipFloats, $tag") { _ =>
        val r = VectorizedAlpValuesReader.forFloat()
        r.initFromPage(NUM_ROWS, ByteBufferInputStream.wrap(ByteBuffer.wrap(bytes)))
        r.skipFloats(NUM_ROWS)
      }
    }
    benchmark.run()
  }

  // --------------- Group C: Single-value reads ---------------

  private def runSingleValueBenchmark(): Unit = {
    val benchmark = new Benchmark(
      "ALP single-value reads", NUM_ROWS.toLong, NUM_ITERS, output = output)

    val rng = new Random(42)

    // Use monetary distribution - representative of typical ALP workloads
    val floatValues = generateFloatData("monetary", NUM_ROWS, rng)
    val doubleValues = generateDoubleData("monetary", NUM_ROWS, rng)
    val floatBytes = encodeAlpFloats(floatValues)
    val doubleBytes = encodeAlpDoubles(doubleValues)

    benchmark.addCase("readFloat (single-value)") { _ =>
      val r = VectorizedAlpValuesReader.forFloat()
      r.initFromPage(NUM_ROWS, ByteBufferInputStream.wrap(ByteBuffer.wrap(floatBytes)))
      var i = 0; while (i < NUM_ROWS) { r.readFloat(); i += 1 }
    }

    benchmark.addCase("readDouble (single-value)") { _ =>
      val r = VectorizedAlpValuesReader.forDouble()
      r.initFromPage(NUM_ROWS, ByteBufferInputStream.wrap(ByteBuffer.wrap(doubleBytes)))
      var i = 0; while (i < NUM_ROWS) { r.readDouble(); i += 1 }
    }

    // Interleaved skip+read pattern simulating predicate pushdown / sparse reads
    benchmark.addCase("skip+read interleaved (float)") { _ =>
      val r = VectorizedAlpValuesReader.forFloat()
      r.initFromPage(NUM_ROWS, ByteBufferInputStream.wrap(ByteBuffer.wrap(floatBytes)))
      var i = 0
      while (i < NUM_ROWS - 1) {
        r.skipFloats(1)
        r.readFloat()
        i += 2
      }
    }

    benchmark.addCase("skip+read interleaved (double)") { _ =>
      val r = VectorizedAlpValuesReader.forDouble()
      r.initFromPage(NUM_ROWS, ByteBufferInputStream.wrap(ByteBuffer.wrap(doubleBytes)))
      var i = 0
      while (i < NUM_ROWS - 1) {
        r.skipDoubles(1)
        r.readDouble()
        i += 2
      }
    }

    benchmark.run()
  }

  override def runBenchmarkSuite(mainArgs: Array[String]): Unit = {
    runBenchmark("ALP DOUBLE decode") { runAlpDoubleBenchmark() }
    runBenchmark("ALP FLOAT decode") { runAlpFloatBenchmark() }
    runBenchmark("ALP single-value reads") { runSingleValueBenchmark() }
  }
}
