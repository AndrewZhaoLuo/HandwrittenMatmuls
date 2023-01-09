/* calculates matmul 512, 3072 @ 3072, 768 -> 512, 768

a0: naive program and measurement, basic outlines
*/
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define REPEATS 100
#define EPS 1e-2

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

/** DRIVER CODE HERE **/
void fill_random(float *mat, int n) {
  for (int i = 0; i < n; i++) {
    mat[i] = (rand() % 1000) / 75.0;
  }
}

void fill_zero(float *mat, int n) {
  for (int i = 0; i < n; i++) {
    mat[i] = 0;
  }
}

bool is_same(float *arr1, float *arr2, int n) {
  for (int i = 0; i < n; i++) {
    float err = fabs(arr1[i] - arr2[i]) / fabs(arr2[i]);
    if (err >= EPS) {
      return false;
    }
  }

  return true;
}

/** YOUR CODE HERE **/

// A, matrix of shape 512, 3072
// B, matrix of shape 768, 3072
// C, matrix of shape 512, 768

template <int registers, int vec_width, int outputs>
void peak_flops(float *A, float *B, int iterations) {
  // A is linearized [registers][vec_width], B is linearized
  // [outputs][vec_width]
  for (int i = 0; i < iterations; i++) {
    for (int r = 0; r < registers; r++) {
      for (int o = 0; o < outputs; o++) {
        for (int v = 0; v < vec_width; v++) {
          B[o * vec_width + v] += A[r * vec_width + v] * B[o * vec_width + v];
        }
      }
    }
  }
}

// baseline matvec multiplication
void matvec_1(float *A, float *B, float *C, int M, int K) {
  for (int m = 0; m < M; m++) {
    float val = 0;
    for (int k = 0; k < K; k++) {
      val += A[m * K + k] * B[k];
    }
    C[m] = val;
  }
}

// block inner reduction
template <int k> void matvec_2(float *A, float *B, float *C, int M, int K) {
  for (int m = 0; m < M; m++) {
    C[m] = 0;
  }

  for (int outer_k = 0; outer_k < K / k; outer_k++) {
    for (int m = 0; m < M; m++) {
      for (int inner_k = 0; inner_k < k; inner_k++) {
        int k_total = outer_k * k + inner_k;
        C[m] += A[m * K + k_total] * B[k_total];
      }
    }
  }

  for (int m = 0; m < M; m++) {
    for (int remain_k = 0; remain_k < K % k; remain_k++) {
      int k_total = K / k * k + remain_k;
      C[m] += A[m * K + k_total] * B[k_total];
    }
  }
}

template <long long vec_width = 8, long long registers = 1024,
          long long outputs = 32>
experiment_result_t measure_peak_flops_avx2(int iterations) {

  float *A = (float *)aligned_alloc(16 * sizeof(float),
                                    vec_width * registers * sizeof(float));
  float *B =
      (float *)aligned_alloc(16 * sizeof(float), vec_width * sizeof(float));

  fill_zero(A, vec_width * registers);
  fill_zero(B, vec_width);

  clock_t start, end;

  // warmup
  peak_flops<registers, vec_width, outputs>(A, B, iterations);

  start = clock();
  peak_flops<registers, vec_width, outputs>(A, B, iterations);
  end = clock();

  free(A);
  free(B);

  double cpu_time_ms = (double)(end - start) / CLOCKS_PER_SEC / 10 * 1e3;
  long long flop = 2 * vec_width * registers * iterations * outputs;
  double flops = (double)flop / (cpu_time_ms / 1e3);
  experiment_result_t results;
  results = (experiment_result_t){end - start, cpu_time_ms, flop, flops};
  return results;
}

experiment_result_t measure_condition(int M, int K, int repeats,
                                      void (*function)(float *, float *,
                                                       float *, int, int)) {
  clock_t start, end;

  float *A = (float *)aligned_alloc(16 * sizeof(float), M * K * sizeof(float));
  float *B = (float *)aligned_alloc(16 * sizeof(float), K * sizeof(float));
  float *C = (float *)aligned_alloc(16 * sizeof(float), M * sizeof(float));

  // warmup
  for (int i = 0; i < repeats; i++) {
    function(A, B, C, M, K);
  }

  start = clock();
  for (int i = 0; i < repeats; i++) {
    function(A, B, C, M, K);
  }
  end = clock();

  free(A);
  free(B);
  free(C);

  double cpu_time_ms = (double)(end - start) / CLOCKS_PER_SEC / 10 * 1e3;
  int flop = M * K * 2 * repeats;
  double flops = (double)flop / (cpu_time_ms / 1e3);

  experiment_result_t results;
  results = (experiment_result_t){end - start, cpu_time_ms, flop, flops};
  return results;
}

int main(int argc, char **argv) {
  if (argc <= 1) {
    printf("usage: ./run function_num\n");
    return 0;
  }

  int function_num = atoi(argv[1]);

  experiment_result_t result;
  switch (function_num) {
  case 0:
    // get peak flops
    result = measure_peak_flops_avx2(1024);
    break;
  case 1:
    result = measure_condition(10000, 1000, 100, matvec_1);
    ;
    break;
  case 2:
    result = measure_condition(10000, 1000, 100, matvec_2<21>);
    ;
    break;
  default:
    printf("Unrecognized function number %d", function_num);
    return 0;
  }

  print_experiment_result(result);

  return 1;
}