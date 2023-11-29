#include "matrix.h"
#include <iostream>

#ifndef __NAIVE_MATMUL_H__
#define __NAIVE_MATMUL_H__

using support::Matrix;

namespace algo {
namespace naive {

// TODO: switch N/M, M comes before M lol
template <typename T>
inline void verifyMatmul(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C, uint32_t &N,
                         uint32_t &M, uint32_t &K) {
  // Checks dimension of matricies assuming we are doing
  // A @ B --> C
  // A is shape [N, K]
  // B is shape [K, M]
  // C is shape [N, M]
  N = A.get_height();
  M = B.get_width();
  K = A.get_width();

  // TODO, add optional assertions controlled by DEBUG flag
}

// STEP 1: The simplest implementation
template <typename T>
void naive_matmul_ijk(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C) {
  uint32_t N, M, K;
  verifyMatmul(A, B, C, N, M, K);

  for (uint32_t i = 0; i < N; i++) {
    for (uint32_t j = 0; j < M; j++) {
      // TODO: possibly make this default constructor of T
      C.a(i, j) = 0;
      for (uint32_t k = 0; k < K; k++) {
        C.a(i, j) += A.r(i, k) * B.r(k, j);
      }
    }
  }
}

// STEP 2: Reordering loops leads to better memory reuse inner loop
// only advances pointers in B.
template <typename T>
void naive_matmul_kij(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C) {
  uint32_t N, M, K;
  verifyMatmul(A, B, C, N, M, K);

  // TODO: ensures this becomes an optimized memset in most cases
  for (uint32_t i = 0; i < N; i++) {
    for (uint32_t j = 0; j < M; j++) {
      C.a(i, j) = 0;
    }
  }

  for (uint32_t k = 0; k < K; k++) {
    for (uint32_t i = 0; i < N; i++) {
      for (uint32_t j = 0; j < M; j++) {
        // TODO: possibly make this default constructor of T
        C.a(i, j) += A.r(i, k) * B.r(k, j);
      }
    }
  }
}

// STEP 3: Rudimentary tiling helps with additional memory improvements
template <typename T, size_t tileN, size_t tileM, size_t tileK>
void tiled_ijk_matmul_ijk(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C) {
  uint32_t N, M, K;
  verifyMatmul(A, B, C, N, M, K);
  T tile_buffer[tileN][tileM];

  // TODO: assertions on divisibility of tiling for now
  for (uint32_t tile_i = 0; tile_i < N / tileN; tile_i++) {
    for (uint32_t tile_j = 0; tile_j < M / tileM; tile_j++) {
      for (uint32_t tile_k = 0; tile_k < K / tileK; tile_k++) {

        // Zero tile_buffer
        for (uint32_t i = 0; i < tileN; i++) {
          for (uint32_t j = 0; j < tileM; j++) {
            tile_buffer[i][j] = 0;
          }
        }

        // ijk
        for (uint32_t inner_i = 0; inner_i < tileN; inner_i++) {
          for (uint32_t inner_j = 0; inner_j < tileM; inner_j++) {
            for (uint32_t inner_k = 0; inner_k < tileK; inner_k++) {
              uint32_t i = tile_i * tileN + inner_i;
              uint32_t j = tile_j * tileM + inner_j;
              uint32_t k = tile_k * tileK + inner_k;
              tile_buffer[inner_i][inner_j] += A.r(i, k) * B.r(k, j);
            }
          }
        }

        // Write buffer back
        for (uint32_t inner_i = 0; inner_i < tileN; inner_i++) {
          for (uint32_t inner_j = 0; inner_j < tileM; inner_j++) {
            uint32_t i = tile_i * tileN + inner_i;
            uint32_t j = tile_j * tileM + inner_j;
            C.a(i, j) = tile_buffer[inner_i][inner_j];
          }
        }
      }
    }
  }
}

// STEP 4: As STEP 3, but with kij inner kernel we know is faster
template <typename T, size_t tileN, size_t tileM, size_t tileK>
void tiled_ijk_matmul_kij(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C) {
  uint32_t N, M, K;
  verifyMatmul(A, B, C, N, M, K);
  T tile_buffer[tileN][tileM];

  // TODO: assertions on divisibility of tiling for now
  for (uint32_t tile_i = 0; tile_i < N / tileN; tile_i++) {
    for (uint32_t tile_j = 0; tile_j < M / tileM; tile_j++) {
      for (uint32_t tile_k = 0; tile_k < K / tileK; tile_k++) {

        // Zero tile_buffer
        for (uint32_t i = 0; i < tileN; i++) {
          for (uint32_t j = 0; j < tileM; j++) {
            tile_buffer[i][j] = 0;
          }
        }

        // kij
        for (uint32_t inner_k = 0; inner_k < tileK; inner_k++) {
          for (uint32_t inner_i = 0; inner_i < tileN; inner_i++) {
            for (uint32_t inner_j = 0; inner_j < tileM; inner_j++) {
              uint32_t i = tile_i * tileN + inner_i;
              uint32_t j = tile_j * tileM + inner_j;
              uint32_t k = tile_k * tileK + inner_k;
              tile_buffer[inner_i][inner_j] += A.r(i, k) * B.r(k, j);
            }
          }
        }

        // Write buffer back
        for (uint32_t inner_i = 0; inner_i < tileN; inner_i++) {
          for (uint32_t inner_j = 0; inner_j < tileM; inner_j++) {
            uint32_t i = tile_i * tileN + inner_i;
            uint32_t j = tile_j * tileM + inner_j;
            C.a(i, j) = tile_buffer[inner_i][inner_j];
          }
        }
      }
    }
  }
}

} // namespace naive
} // namespace algo
#endif