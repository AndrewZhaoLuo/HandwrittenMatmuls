#include "matrix.h"
#include <iostream>

#ifndef __NAIVE_MATMUL_H__
#define __NAIVE_MATMUL_H__

using support::Matrix;

namespace algo {
namespace naive {

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

} // namespace naive
} // namespace algo
#endif