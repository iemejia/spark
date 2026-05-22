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
package org.apache.spark.sql.execution.datasources.parquet;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.apache.parquet.bytes.ByteBufferInputStream;
import org.apache.parquet.column.values.bitpacking.BytePacker;
import org.apache.parquet.column.values.bitpacking.BytePackerForLong;
import org.apache.parquet.column.values.bitpacking.Packer;
import org.apache.parquet.io.ParquetDecodingException;

import org.apache.spark.sql.execution.vectorized.WritableColumnVector;

/**
 * Vectorized reader for ALP (Adaptive Lossless floating-Point) encoded Parquet pages.
 *
 * <p>ALP encoding converts floating-point values to integers using decimal scaling,
 * applies Frame of Reference (FOR) encoding, and bit-packs the deltas. Values that
 * cannot be losslessly converted are stored as exceptions with their original raw bits.
 *
 * <p>This reader supports both FLOAT and DOUBLE types and decodes data in vectors
 * (default 1024 values), matching the ALP page layout:
 * <pre>
 * [Header 7B]: compressionMode(1) + integerEncoding(1) + logVectorSize(1) + numElements(4)
 * [Offset Array]: 4B per vector (byte offset to vector data)
 * [Vector 0..N]:
 *   AlpInfo(4B): exponent(1) + factor(1) + numExceptions(2 LE)
 *   ForInfo(5B float / 9B double): frameOfReference(4/8 LE) + bitWidth(1)
 *   PackedValues: bit-packed deltas (groups of 8)
 *   ExceptionPositions: uint16 LE per exception
 *   ExceptionValues: raw float(4B) / double(8B) LE per exception
 * </pre>
 *
 * <p>Based on the paper: "ALP: Adaptive Lossless floating-Point Compression" (SIGMOD 2024)
 *
 * @see <a href="https://dl.acm.org/doi/10.1145/3626717">ALP Paper</a>
 */
public class VectorizedAlpValuesReader extends VectorizedReaderBase {

  // ALP page header constants
  private static final int ALP_HEADER_SIZE = 7;
  private static final int ALP_INFO_SIZE = 4;
  private static final int FLOAT_FOR_INFO_SIZE = 5;
  private static final int DOUBLE_FOR_INFO_SIZE = 9;
  private static final int ALP_COMPRESSION_MODE = 0;
  private static final int ALP_INTEGER_ENCODING_FOR = 0;
  private static final int MIN_LOG_VECTOR_SIZE = 3;
  private static final int MAX_LOG_VECTOR_SIZE = 15;

  // Powers of 10 for float encode/decode
  private static final float[] FLOAT_POW10 = {
    1e0f, 1e1f, 1e2f, 1e3f, 1e4f, 1e5f, 1e6f, 1e7f, 1e8f, 1e9f, 1e10f
  };
  private static final float[] FLOAT_POW10_NEGATIVE = {
    1e0f, 1e-1f, 1e-2f, 1e-3f, 1e-4f, 1e-5f, 1e-6f, 1e-7f, 1e-8f, 1e-9f, 1e-10f
  };

  // Powers of 10 for double encode/decode
  private static final double[] DOUBLE_POW10 = {
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
    1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18
  };
  private static final double[] DOUBLE_POW10_NEGATIVE = {
    1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
    1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18
  };

  // Page-level state
  private int vectorSize;
  private int vectorMask;
  private int logVectorSize;
  private int totalCount;
  private int numVectors;
  private int[] vectorOffsets;
  private ByteBuffer vectorsData;
  private int offsetArraySize;

  // Backing byte array for vectorsData (null if direct buffer).
  // Using byte[] with the generated packer's array overload avoids ByteBuffer bounds-checking
  // overhead in the unpack hot path.
  private byte[] vectorsArray;
  private int vectorsArrayBase;

  // Current position
  private int currentIndex;

  // Decoded vector buffer (lazily decoded)
  private int currentVectorIndex = -1;
  private float[] decodedFloats;
  private double[] decodedDoubles;

  // Whether this reader is being used for float or double
  private boolean isFloat;

  // Reusable unpack buffers
  private int[] intDeltasBuffer;
  private long[] longDeltasBuffer;
  private int[] excPositionsBuffer;
  private final int[] intUnpackPadBuf = new int[8];
  private final long[] longUnpackPadBuf = new long[8];
  private byte[] unpackByteBuf;

  /**
   * Creates an ALP reader for FLOAT type.
   */
  public static VectorizedAlpValuesReader forFloat() {
    VectorizedAlpValuesReader reader = new VectorizedAlpValuesReader();
    reader.isFloat = true;
    return reader;
  }

  /**
   * Creates an ALP reader for DOUBLE type.
   */
  public static VectorizedAlpValuesReader forDouble() {
    VectorizedAlpValuesReader reader = new VectorizedAlpValuesReader();
    reader.isFloat = false;
    return reader;
  }

  public VectorizedAlpValuesReader() {
    this.currentIndex = 0;
  }

  @Override
  public void initFromPage(int valueCount, ByteBufferInputStream stream) throws IOException {
    // Read the 7-byte header
    ByteBuffer headerBuf = stream.slice(ALP_HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN);
    int compressionMode = headerBuf.get() & 0xFF;
    int integerEncoding = headerBuf.get() & 0xFF;
    int logVecSize = headerBuf.get() & 0xFF;
    int numElements = headerBuf.getInt();

    if (compressionMode != ALP_COMPRESSION_MODE) {
      throw new ParquetDecodingException("Unsupported ALP compression mode: " + compressionMode);
    }
    if (integerEncoding != ALP_INTEGER_ENCODING_FOR) {
      throw new ParquetDecodingException("Unsupported ALP integer encoding: " + integerEncoding);
    }
    if (logVecSize < MIN_LOG_VECTOR_SIZE || logVecSize > MAX_LOG_VECTOR_SIZE) {
      throw new ParquetDecodingException(
          "Invalid ALP log vector size: " + logVecSize +
          ", must be between " + MIN_LOG_VECTOR_SIZE + " and " + MAX_LOG_VECTOR_SIZE);
    }
    if (numElements < 0) {
      throw new ParquetDecodingException("Invalid ALP element count: " + numElements);
    }

    this.vectorSize = 1 << logVecSize;
    this.vectorMask = this.vectorSize - 1;
    this.logVectorSize = logVecSize;
    this.totalCount = numElements;
    this.numVectors = (numElements + vectorSize - 1) / vectorSize;
    this.currentIndex = 0;
    this.currentVectorIndex = -1;

    // Read the offset array
    this.offsetArraySize = numVectors * Integer.BYTES;
    ByteBuffer offsetBuf = stream.slice(offsetArraySize).order(ByteOrder.LITTLE_ENDIAN);
    this.vectorOffsets = new int[numVectors];
    for (int v = 0; v < numVectors; v++) {
      vectorOffsets[v] = offsetBuf.getInt();
    }

    // Slice remaining bytes as the vectors data buffer (0-based positioning)
    int remainingBytes = (int) stream.available();
    ByteBuffer rawSlice = stream.slice(remainingBytes);
    this.vectorsData = rawSlice.slice().order(ByteOrder.LITTLE_ENDIAN);

    // Extract backing array for faster unpack access (avoids ByteBuffer bounds checking)
    if (this.vectorsData.hasArray()) {
      this.vectorsArray = this.vectorsData.array();
      this.vectorsArrayBase = this.vectorsData.arrayOffset() + this.vectorsData.position();
    } else {
      this.vectorsArray = null;
      this.vectorsArrayBase = 0;
    }

    // Allocate decode buffers
    allocateBuffers();
  }

  private void allocateBuffers() {
    if (isFloat) {
      this.decodedFloats = new float[vectorSize];
      this.intDeltasBuffer = new int[vectorSize];
      // max bit width for int packing = 32 bytes per group of 8
      this.unpackByteBuf = new byte[Integer.SIZE];
    } else {
      this.decodedDoubles = new double[vectorSize];
      this.longDeltasBuffer = new long[vectorSize];
      // max bit width for long packing = 64 bytes per group of 8
      this.unpackByteBuf = new byte[Long.SIZE];
    }
    this.excPositionsBuffer = new int[vectorSize];
  }

  // ======================== Single-value reads ========================

  @Override
  public float readFloat() {
    if (currentIndex >= totalCount) {
      throw new ParquetDecodingException("ALP float data exhausted at index " + currentIndex);
    }
    ensureVectorDecoded();
    int indexInVector = currentIndex & vectorMask;
    currentIndex++;
    return decodedFloats[indexInVector];
  }

  @Override
  public double readDouble() {
    if (currentIndex >= totalCount) {
      throw new ParquetDecodingException("ALP double data exhausted at index " + currentIndex);
    }
    ensureVectorDecoded();
    int indexInVector = currentIndex & vectorMask;
    currentIndex++;
    return decodedDoubles[indexInVector];
  }

  // ======================== Batch reads ========================

  @Override
  public void readFloats(int total, WritableColumnVector c, int rowId) {
    int remaining = total;
    while (remaining > 0) {
      ensureVectorDecoded();
      int indexInVector = currentIndex & vectorMask;
      int currentVectorLen = getVectorLength(currentIndex >>> logVectorSize);
      int availableInVector = currentVectorLen - indexInVector;
      int toRead = Math.min(remaining, availableInVector);

      c.putFloats(rowId, toRead, decodedFloats, indexInVector);

      currentIndex += toRead;
      rowId += toRead;
      remaining -= toRead;
    }
  }

  @Override
  public void readDoubles(int total, WritableColumnVector c, int rowId) {
    int remaining = total;
    while (remaining > 0) {
      ensureVectorDecoded();
      int indexInVector = currentIndex & vectorMask;
      int currentVectorLen = getVectorLength(currentIndex >>> logVectorSize);
      int availableInVector = currentVectorLen - indexInVector;
      int toRead = Math.min(remaining, availableInVector);

      c.putDoubles(rowId, toRead, decodedDoubles, indexInVector);

      currentIndex += toRead;
      rowId += toRead;
      remaining -= toRead;
    }
  }

  @Override
  public void skipFloats(int total) {
    skipValues(total);
  }

  @Override
  public void skipDoubles(int total) {
    skipValues(total);
  }

  @Override
  public void skip() {
    skipValues(1);
  }

  private void skipValues(int n) {
    if (n < 0 || currentIndex + n > totalCount) {
      throw new ParquetDecodingException(String.format(
          "Cannot skip %d ALP values. Current index: %d, total count: %d",
          n, currentIndex, totalCount));
    }
    currentIndex += n;
  }

  // ======================== Internal decoding ========================

  private void ensureVectorDecoded() {
    int vectorIdx = currentIndex >>> logVectorSize;
    if (vectorIdx != currentVectorIndex) {
      if (isFloat) {
        decodeFloatVector(vectorIdx);
      } else {
        decodeDoubleVector(vectorIdx);
      }
      currentVectorIndex = vectorIdx;
    }
  }

  private int getVectorLength(int vectorIdx) {
    if (vectorIdx < numVectors - 1) {
      return vectorSize;
    }
    int lastVectorLen = totalCount & vectorMask;
    return lastVectorLen == 0 ? vectorSize : lastVectorLen;
  }

  private int getVectorDataPosition(int vectorIdx) {
    return vectorOffsets[vectorIdx] - offsetArraySize;
  }

  /**
   * Decode a float vector from the ALP page.
   * Layout per vector: AlpInfo(4B) + ForInfo(5B) + PackedValues + ExcPositions + ExcValues
   */
  private void decodeFloatVector(int vectorIdx) {
    int vectorLen = getVectorLength(vectorIdx);
    int pos = getVectorDataPosition(vectorIdx);

    // AlpInfo: exponent(1) + factor(1) + numExceptions(2 LE)
    int exponent = vectorsData.get(pos) & 0xFF;
    int factor = vectorsData.get(pos + 1) & 0xFF;
    int numExceptions = vectorsData.getShort(pos + 2) & 0xFFFF;
    pos += ALP_INFO_SIZE;

    // ForInfo: frameOfReference(4 LE) + bitWidth(1)
    int frameOfReference = vectorsData.getInt(pos);
    int bitWidth = vectorsData.get(pos + 4) & 0xFF;
    pos += FLOAT_FOR_INFO_SIZE;

    // Unpack bit-packed deltas and decode
    float pow10f = FLOAT_POW10[factor];
    float pow10ne = FLOAT_POW10_NEGATIVE[exponent];
    if (bitWidth > 0) {
      pos = unpackInts(vectorsData, pos, intDeltasBuffer, vectorLen, bitWidth);
      for (int i = 0; i < vectorLen; i++) {
        decodedFloats[i] = (intDeltasBuffer[i] + frameOfReference) * pow10f * pow10ne;
      }
    } else {
      // bitWidth=0: all deltas are zero, all values decode to the same result
      float value = frameOfReference * pow10f * pow10ne;
      java.util.Arrays.fill(decodedFloats, 0, vectorLen, value);
    }

    // Apply exceptions: overwrite positions with raw float values
    if (numExceptions > 0) {
      for (int e = 0; e < numExceptions; e++) {
        excPositionsBuffer[e] = vectorsData.getShort(pos) & 0xFFFF;
        pos += Short.BYTES;
      }
      for (int e = 0; e < numExceptions; e++) {
        decodedFloats[excPositionsBuffer[e]] = Float.intBitsToFloat(vectorsData.getInt(pos));
        pos += Float.BYTES;
      }
    }
  }

  /**
   * Decode a double vector from the ALP page.
   * Layout per vector: AlpInfo(4B) + ForInfo(9B) + PackedValues + ExcPositions + ExcValues
   */
  private void decodeDoubleVector(int vectorIdx) {
    int vectorLen = getVectorLength(vectorIdx);
    int pos = getVectorDataPosition(vectorIdx);

    // AlpInfo: exponent(1) + factor(1) + numExceptions(2 LE)
    int exponent = vectorsData.get(pos) & 0xFF;
    int factor = vectorsData.get(pos + 1) & 0xFF;
    int numExceptions = vectorsData.getShort(pos + 2) & 0xFFFF;
    pos += ALP_INFO_SIZE;

    // ForInfo: frameOfReference(8 LE) + bitWidth(1)
    long frameOfReference = vectorsData.getLong(pos);
    int bitWidth = vectorsData.get(pos + 8) & 0xFF;
    pos += DOUBLE_FOR_INFO_SIZE;

    // Unpack bit-packed deltas and decode
    double pow10f = DOUBLE_POW10[factor];
    double pow10ne = DOUBLE_POW10_NEGATIVE[exponent];
    if (bitWidth > 0) {
      pos = unpackLongs(vectorsData, pos, longDeltasBuffer, vectorLen, bitWidth);
      for (int i = 0; i < vectorLen; i++) {
        decodedDoubles[i] = (longDeltasBuffer[i] + frameOfReference) * pow10f * pow10ne;
      }
    } else {
      // bitWidth=0: all deltas are zero, all values decode to the same result
      double value = frameOfReference * pow10f * pow10ne;
      java.util.Arrays.fill(decodedDoubles, 0, vectorLen, value);
    }

    // Apply exceptions: overwrite positions with raw double values
    if (numExceptions > 0) {
      for (int e = 0; e < numExceptions; e++) {
        excPositionsBuffer[e] = vectorsData.getShort(pos) & 0xFFFF;
        pos += Short.BYTES;
      }
      for (int e = 0; e < numExceptions; e++) {
        decodedDoubles[excPositionsBuffer[e]] = Double.longBitsToDouble(vectorsData.getLong(pos));
        pos += Double.BYTES;
      }
    }
  }

  // ======================== Bit unpacking ========================

  /**
   * Unpack bit-packed int values using unpack32Values for the bulk and unpack8Values for the tail.
   * When a backing byte array is available, uses the byte[] overload to avoid ByteBuffer
   * bounds-checking overhead in the hot loop.
   * Returns the position after all packed data.
   */
  private int unpackInts(ByteBuffer buf, int pos, int[] output, int count, int bitWidth) {
    BytePacker packer = Packer.LITTLE_ENDIAN.newBytePacker(bitWidth);
    int numGroups32 = count / 32;
    int remaining = count - numGroups32 * 32;

    if (vectorsArray != null) {
      int arrayPos = vectorsArrayBase + pos;

      // Process 32 values at a time using byte[] overload
      for (int g = 0; g < numGroups32; g++) {
        packer.unpack32Values(vectorsArray, arrayPos, output, g * 32);
        arrayPos += bitWidth * 4;
      }

      // Process remaining in groups of 8
      int offset = numGroups32 * 32;
      int numGroups8 = remaining / 8;
      int tail = remaining - numGroups8 * 8;

      for (int g = 0; g < numGroups8; g++) {
        packer.unpack8Values(vectorsArray, arrayPos, output, offset + g * 8);
        arrayPos += bitWidth;
      }

      if (tail > 0) {
        int totalPackedBytes = (count * bitWidth + 7) / 8;
        int alreadyRead = (numGroups32 * 4 + numGroups8) * bitWidth;
        int partialBytes = totalPackedBytes - alreadyRead;

        System.arraycopy(vectorsArray, arrayPos, unpackByteBuf, 0, partialBytes);
        for (int i = partialBytes; i < bitWidth; i++) {
          unpackByteBuf[i] = 0;
        }

        packer.unpack8Values(unpackByteBuf, 0, intUnpackPadBuf, 0);
        System.arraycopy(intUnpackPadBuf, 0, output, offset + numGroups8 * 8, tail);
        arrayPos += partialBytes;
      }

      pos = arrayPos - vectorsArrayBase;
    } else {
      // Fallback: ByteBuffer path for direct buffers
      for (int g = 0; g < numGroups32; g++) {
        packer.unpack32Values(buf, pos, output, g * 32);
        pos += bitWidth * 4;
      }

      int offset = numGroups32 * 32;
      int numGroups8 = remaining / 8;
      int tail = remaining - numGroups8 * 8;

      for (int g = 0; g < numGroups8; g++) {
        packer.unpack8Values(buf, pos, output, offset + g * 8);
        pos += bitWidth;
      }

      if (tail > 0) {
        int totalPackedBytes = (count * bitWidth + 7) / 8;
        int alreadyRead = (numGroups32 * 4 + numGroups8) * bitWidth;
        int partialBytes = totalPackedBytes - alreadyRead;

        for (int i = 0; i < partialBytes; i++) {
          unpackByteBuf[i] = buf.get(pos + i);
        }
        for (int i = partialBytes; i < bitWidth; i++) {
          unpackByteBuf[i] = 0;
        }

        packer.unpack8Values(unpackByteBuf, 0, intUnpackPadBuf, 0);
        System.arraycopy(intUnpackPadBuf, 0, output, offset + numGroups8 * 8, tail);
        pos += partialBytes;
      }
    }

    return pos;
  }

  /**
   * Unpack bit-packed long values using BytePackerForLong.
   * For narrow bit-widths (<=32), uses unpack32Values for better throughput.
   * For wide bit-widths (>32), uses unpack8Values which performs better due to
   * reduced per-call memory traffic in the ByteBuffer read path.
   * When a backing byte array is available, uses the byte[] overload.
   * Returns the position after all packed data.
   */
  private int unpackLongs(ByteBuffer buf, int pos, long[] output, int count, int bitWidth) {
    BytePackerForLong packer = Packer.LITTLE_ENDIAN.newBytePackerForLong(bitWidth);

    if (vectorsArray != null) {
      int arrayPos = vectorsArrayBase + pos;

      if (bitWidth <= 32) {
        int numGroups32 = count / 32;
        int remaining = count - numGroups32 * 32;

        for (int g = 0; g < numGroups32; g++) {
          packer.unpack32Values(vectorsArray, arrayPos, output, g * 32);
          arrayPos += bitWidth * 4;
        }

        int offset = numGroups32 * 32;
        int numGroups8 = remaining / 8;
        int tail = remaining - numGroups8 * 8;

        for (int g = 0; g < numGroups8; g++) {
          packer.unpack8Values(vectorsArray, arrayPos, output, offset + g * 8);
          arrayPos += bitWidth;
        }

        if (tail > 0) {
          int totalPackedBytes = (count * bitWidth + 7) / 8;
          int alreadyRead = (numGroups32 * 4 + numGroups8) * bitWidth;
          int partialBytes = totalPackedBytes - alreadyRead;

          System.arraycopy(vectorsArray, arrayPos, unpackByteBuf, 0, partialBytes);
          for (int i = partialBytes; i < bitWidth; i++) {
            unpackByteBuf[i] = 0;
          }

          packer.unpack8Values(unpackByteBuf, 0, longUnpackPadBuf, 0);
          System.arraycopy(longUnpackPadBuf, 0, output, offset + numGroups8 * 8, tail);
          arrayPos += partialBytes;
        }
      } else {
        int numFullGroups = count / 8;
        int remaining = count % 8;

        for (int g = 0; g < numFullGroups; g++) {
          packer.unpack8Values(vectorsArray, arrayPos, output, g * 8);
          arrayPos += bitWidth;
        }

        if (remaining > 0) {
          int totalPackedBytes = (count * bitWidth + 7) / 8;
          int alreadyRead = numFullGroups * bitWidth;
          int partialBytes = totalPackedBytes - alreadyRead;

          System.arraycopy(vectorsArray, arrayPos, unpackByteBuf, 0, partialBytes);
          for (int i = partialBytes; i < bitWidth; i++) {
            unpackByteBuf[i] = 0;
          }

          packer.unpack8Values(unpackByteBuf, 0, longUnpackPadBuf, 0);
          System.arraycopy(longUnpackPadBuf, 0, output, numFullGroups * 8, remaining);
          arrayPos += partialBytes;
        }
      }

      pos = arrayPos - vectorsArrayBase;
    } else {
      // Fallback: ByteBuffer path for direct buffers
      if (bitWidth <= 32) {
        int numGroups32 = count / 32;
        int remaining = count - numGroups32 * 32;

        for (int g = 0; g < numGroups32; g++) {
          packer.unpack32Values(buf, pos, output, g * 32);
          pos += bitWidth * 4;
        }

        int offset = numGroups32 * 32;
        int numGroups8 = remaining / 8;
        int tail = remaining - numGroups8 * 8;

        for (int g = 0; g < numGroups8; g++) {
          packer.unpack8Values(buf, pos, output, offset + g * 8);
          pos += bitWidth;
        }

        if (tail > 0) {
          int totalPackedBytes = (count * bitWidth + 7) / 8;
          int alreadyRead = (numGroups32 * 4 + numGroups8) * bitWidth;
          int partialBytes = totalPackedBytes - alreadyRead;

          for (int i = 0; i < partialBytes; i++) {
            unpackByteBuf[i] = buf.get(pos + i);
          }
          for (int i = partialBytes; i < bitWidth; i++) {
            unpackByteBuf[i] = 0;
          }

          packer.unpack8Values(unpackByteBuf, 0, longUnpackPadBuf, 0);
          System.arraycopy(longUnpackPadBuf, 0, output, offset + numGroups8 * 8, tail);
          pos += partialBytes;
        }
      } else {
        int numFullGroups = count / 8;
        int remaining = count % 8;

        for (int g = 0; g < numFullGroups; g++) {
          packer.unpack8Values(buf, pos, output, g * 8);
          pos += bitWidth;
        }

        if (remaining > 0) {
          int totalPackedBytes = (count * bitWidth + 7) / 8;
          int alreadyRead = numFullGroups * bitWidth;
          int partialBytes = totalPackedBytes - alreadyRead;

          for (int i = 0; i < partialBytes; i++) {
            unpackByteBuf[i] = buf.get(pos + i);
          }
          for (int i = partialBytes; i < bitWidth; i++) {
            unpackByteBuf[i] = 0;
          }

          packer.unpack8Values(unpackByteBuf, 0, longUnpackPadBuf, 0);
          System.arraycopy(longUnpackPadBuf, 0, output, numFullGroups * 8, remaining);
          pos += partialBytes;
        }
      }
    }

    return pos;
  }
}
