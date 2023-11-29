#include "matrix.h"
#include "naive_matmul.h"
#include <chrono>
#include <functional>
#include <iostream>
#include <random>

using support::Matrix;

using f_matmul_float =
    std::function<void(Matrix<float> &, Matrix<float> &, Matrix<float> &)>;

void randomInitFloatMatrix(Matrix<float> &mat) {
  std::default_random_engine generator(0);
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  for (int r = 0; r < mat.get_height(); r++) {
    for (int c = 0; c < mat.get_width(); c++) {
      mat.a(r, c) = distribution(generator);
    }
  }
}

template <int warmups, int repeats>
int64_t benchmark_get_us(Matrix<float> &matA, Matrix<float> &matB,
                         Matrix<float> &matC, f_matmul_float func) {
  for (int i = 0; i < warmups; i++) {
    func(matA, matB, matC);
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeats; i++) {
    func(matA, matB, matC);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  return us.count() / repeats;
}

template <int M, int N, int K, int warmups, int repeats>
void test_conditions() {
  Matrix<float> matA(M, K);
  Matrix<float> matB(K, N);
  Matrix<float> matC(M, N);

  randomInitFloatMatrix(matA);
  randomInitFloatMatrix(matB);
  randomInitFloatMatrix(matC);

  std::cout << "M = " << M << ", N = " << N << ", K = " << K
            << ", warmups = " << warmups << ", repeats " << repeats
            << std::endl;

  uint64_t us_ijk = benchmark_get_us<warmups, repeats>(
      matA, matB, matC, algo::naive::naive_matmul_ijk<float>);
  std::cout << "\tnaive_matmul_ijk (us): " << us_ijk << std::endl;

  uint64_t us_kij = benchmark_get_us<warmups, repeats>(
      matA, matB, matC, algo::naive::naive_matmul_kij<float>);
  std::cout << "\tnaive_matmul_kij (us): " << us_kij << std::endl;

  uint64_t us_tiled8ijk_inner_ijk = benchmark_get_us<warmups, repeats>(
      matA, matB, matC, algo::naive::tiled_ijk_matmul_ijk<float, 8, 8, 8>);
  std::cout << "\tiled_ijk_matmul_ijk<float, 8, 8, 8> (us): "
            << us_tiled8ijk_inner_ijk << std::endl;

  uint64_t us_tiled16_ijk_inner_ijk = benchmark_get_us<warmups, repeats>(
      matA, matB, matC, algo::naive::tiled_ijk_matmul_ijk<float, 16, 16, 16>);
  std::cout << "\tiled_ijk_matmul_ijk<float, 16, 16, 16> (us): "
            << us_tiled16_ijk_inner_ijk << std::endl;

  uint64_t us_tiled32_ijk_inner_ijk = benchmark_get_us<warmups, repeats>(
      matA, matB, matC, algo::naive::tiled_ijk_matmul_ijk<float, 32, 32, 32>);
  std::cout << "\tiled_ijk_matmul_ijk<float, 32, 32, 32> (us): "
            << us_tiled32_ijk_inner_ijk << std::endl;

  uint64_t us_tiled8_ijk_inner_kij = benchmark_get_us<warmups, repeats>(
      matA, matB, matC, algo::naive::tiled_ijk_matmul_kij<float, 8, 8, 8>);
  std::cout << "\tiled_ijk_matmul_kij<float, 8, 8, 8> (us): "
            << us_tiled8_ijk_inner_kij << std::endl;

  uint64_t us_tiled16_ijk_inner_kij = benchmark_get_us<warmups, repeats>(
      matA, matB, matC, algo::naive::tiled_ijk_matmul_kij<float, 16, 16, 16>);
  std::cout << "\tiled_ijk_matmul_kij<float, 16, 16, 16> (us): "
            << us_tiled16_ijk_inner_kij << std::endl;

  uint64_t us_tiled32_ijk_inner_kij = benchmark_get_us<warmups, repeats>(
      matA, matB, matC, algo::naive::tiled_ijk_matmul_kij<float, 32, 32, 32>);
  std::cout << "\tiled_ijk_matmul_kij<float, 32, 32, 32> (us): "
            << us_tiled32_ijk_inner_kij << std::endl;
}

void example_simple() {
  std::cout << "Hello world!" << std::endl;
  Matrix<float> matA(32, 32);
  Matrix<float> matB(32, 32);
  Matrix<float> matC(32, 32);

  randomInitFloatMatrix(matA);
  randomInitFloatMatrix(matB);
  randomInitFloatMatrix(matC);
  std::cout << "Matrix A: " << std::endl;
  std::cout << matA << std::endl;
  std::cout << "Matrix B: " << std::endl;
  std::cout << matB << std::endl;

  algo::naive::naive_matmul_ijk(matA, matB, matC);
  std::cout << "Matrix C naive inner product: " << std::endl;
  std::cout << matC << std::endl;

  algo::naive::naive_matmul_kij(matA, matB, matC);
  std::cout << "Matrix C naive outer product: " << std::endl;
  std::cout << matC << std::endl;

  algo::naive::tiled_ijk_matmul_ijk<float, 16, 16, 16>(matA, matB, matC);
  std::cout << "Matrix C tiled ijk inner ijk product: " << std::endl;
  std::cout << matC << std::endl;

  algo::naive::tiled_ijk_matmul_kij<float, 16, 16, 16>(matA, matB, matC);
  std::cout << "Matrix C tiled ijk inner kij product: " << std::endl;
  std::cout << matC << std::endl;
}

int main() {
  example_simple();

  test_conditions<256, 256, 256, 1, 10>();
  test_conditions<512, 512, 512, 1, 10>();
  test_conditions<1024, 1024, 1024, 1, 10>();

  return 0;
}