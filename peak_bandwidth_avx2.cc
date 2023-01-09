
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VEC_WIDTH 8
#define ITERATIONS 1024 * 1024

typedef struct experiment_result {
  clock_t clock_cycles;
  double ms;
  long long flop;
  double flops;
} experiment_result_t;

void print_experiment_result(experiment_result result) {
  printf("%.4lf (ms)\n", result.ms);
  printf("%lld flop\n", result.flop);
  printf("%.4le FLOPS\n", result.flops);
}

template <int iterations> void peak_mul_add(float *A, float *B) {
  __m256 a_reg = _mm256_set_ps(A[7], A[6], A[5], A[4], A[3], A[2], A[1], A[0]);
  __m256 b_reg = _mm256_set_ps(B[7], B[6], B[5], B[4], B[3], B[2], B[1], B[0]);

  for (int i = 0; i < iterations; i++) {
    __m256 mul_tmp = _mm256_mul_ps(a_reg, b_reg);
    a_reg = _mm256_add_ps(mul_tmp, a_reg);
  }
  _mm256_storeu_ps(A, a_reg);
}

template <int iterations> void peak_fma(float *A, float *B) {
  __m256 a_reg = _mm256_set_ps(A[7], A[6], A[5], A[4], A[3], A[2], A[1], A[0]);
  __m256 b_reg = _mm256_set_ps(B[7], B[6], B[5], B[4], B[3], B[2], B[1], B[0]);

  for (int i = 0; i < iterations; i++) {
    a_reg = _mm256_fmadd_ps(a_reg, b_reg, a_reg);
  }
  _mm256_storeu_ps(A, a_reg);
}

int main() {
  clock_t start, end;
  float *A =
      (float *)aligned_alloc(16 * sizeof(float), VEC_WIDTH * sizeof(float));
  float *B =
      (float *)aligned_alloc(16 * sizeof(float), VEC_WIDTH * sizeof(float));

  for (int sample = 0; sample < 10; sample++) {
    start = clock();
    // seems to have 50% higher throughput
    peak_fma<ITERATIONS>(A, B);
    end = clock();

    double cpu_time_ms = (double)(end - start) / CLOCKS_PER_SEC / 10 * 1e3;
    long long flop = 2 * VEC_WIDTH * ITERATIONS;
    double flops = (double)flop / (cpu_time_ms / 1e3);
    experiment_result_t results;
    results = (experiment_result_t){end - start, cpu_time_ms, flop, flops};

    print_experiment_result(results);
    printf("\n");
  }
  free(A);
  free(B);
  return 1;
}