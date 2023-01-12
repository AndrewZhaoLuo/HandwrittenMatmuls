/* calculates matmul 512, 3072 @ 3072, 768 -> 512, 768

a0: naive program and measurement, basic outlines
*/
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LIKWID_PERFMON
#include <likwid.h>

#define ALL_FUNCTIONS4(FUNC, OFFSET)                                           \
  &FUNC<0 + OFFSET>, &FUNC<1 + OFFSET>, &FUNC<2 + OFFSET>, &FUNC<3 + OFFSET>
#define ALL_FUNCTIONS8(FUNC, OFFSET)                                           \
  ALL_FUNCTIONS4(FUNC, 0 + OFFSET), ALL_FUNCTIONS4(FUNC, 4 + OFFSET)
#define ALL_FUNCTIONS16(FUNC, OFFSET)                                          \
  ALL_FUNCTIONS8(FUNC, 0 + OFFSET), ALL_FUNCTIONS8(FUNC, 8 + OFFSET)
#define ALL_FUNCTIONS31(FUNC)                                                  \
  ALL_FUNCTIONS16(FUNC, 0), ALL_FUNCTIONS16(FUNC, 16)
#define FTABLE_0_31(FUNC)                                                      \
  { ALL_FUNCTIONS31(FUNC) }

#define REPEATS 10
#define EPS 1e-2

typedef struct experiment_result {
  double ms;
  double flops;
  long long flop;
  clock_t clock_cycles;
  bool correctness;
} experiment_result_t;

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

// peak flops for something with O(MK) flop
template <int param_fma_ops_inner_loop>
void peak_flops(float *A, float *B, float *C, int M, int K) {
  int fma_ops_inner_loop = param_fma_ops_inner_loop;
  if (fma_ops_inner_loop == 0) {
    fma_ops_inner_loop = 1;
  }

  __m256 a1_reg = _mm256_set_ps(A[7], A[6], A[5], A[4], A[3], A[2], A[1], A[0]);
  __m256 b1_reg = _mm256_set_ps(B[7], B[6], B[5], B[4], B[3], B[2], B[1], B[0]);

  __m256 c_regs[fma_ops_inner_loop];
  for (int r = 0; r < fma_ops_inner_loop; r++) {
    c_regs[r] = _mm256_set_ps(0, 0, 0, 0, 0, 0, 0, 0);
  }

  // want total flops to be M * K * 2 to match other matvecs
  // inner loop does 8 * 2 * fma_ops_inner_loop flop so
  // need to run M * K / (8 * fma_ops_inner_loop) times
  long long target_flop = M * K * 2;
  long long flop_per_loop_iter = 8 * 2 * fma_ops_inner_loop;
  int iterations = target_flop / flop_per_loop_iter;
  int remainder = target_flop % flop_per_loop_iter;

  for (int i = 0; i < iterations; i++) {
    // two fma units per core (CPI = 0.5), seems to saturate at 2
    for (int r = 0; r < fma_ops_inner_loop; r++) {
      c_regs[r] = _mm256_fmadd_ps(a1_reg, b1_reg, c_regs[r]);
    }
  }
  // need remainder flops 
  for (int i = 0; i < remainder; i++) {
    C[0] += C[i % M];
  }

  __m256 c_out_reg =
      _mm256_set_ps(C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7]);
  for (int r = 0; r < fma_ops_inner_loop; r++) {
    c_out_reg = _mm256_add_ps(c_out_reg, c_regs[r]);
  }
  _mm256_storeu_ps(C, c_out_reg);
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
template <int param_k> void matvec_2(float *A, float *B, float *C, int M, int K) {
  int k = param_k;
  if (k == 0) {
    k = 1;
  }

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

// from https://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
float sum8(__m256 x) {
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}


// block inner reduction, vectorize
template <int param_num_inner_reduction, int vec_width=8> void matvec_3(float *A, float *B, float *C, int M, int K) {
  int num_inner_reduction = param_num_inner_reduction;
  if (num_inner_reduction == 0) {
    num_inner_reduction = 1;
  }

  __m256 b_regs[num_inner_reduction];
  for (int m = 0; m < M; m++) {
    C[m] = 0;
  } 

  int ele_b_per_outer_loop = vec_width * num_inner_reduction;
  int total_blocks_k = K / ele_b_per_outer_loop;
  int remainder_blocks_k = K % ele_b_per_outer_loop;
  for (int k_block = 0; k_block < total_blocks_k; k_block++) {
    float* B_tmp = B + k_block * ele_b_per_outer_loop;
    for (int inner_load = 0; inner_load < num_inner_reduction; inner_load++) {
      b_regs[inner_load] = _mm256_load_ps(B_tmp);
      B_tmp += vec_width;
    }

    // calculate C[m]_partial
    for (int m = 0; m < M; m++) {
      float* A_tmp = A + m * K + k_block * ele_b_per_outer_loop;
      __m256 c_reg = _mm256_set1_ps(0);
      for (int k = 0; k < num_inner_reduction; k++) {
        __m256 a_reg = _mm256_load_ps(A_tmp);
        A_tmp += vec_width;
        c_reg = _mm256_fmadd_ps(a_reg, b_regs[k], c_reg);
      }
      C[m] += sum8(c_reg);
    }
  }
}


experiment_result_t measure_condition(int M, int K, int repeats,
                                      void (*function)(float *, float *,
                                                       float *, int, int)) {
  clock_t start, end;

  float *A = (float *)aligned_alloc(16 * sizeof(float), M * K * sizeof(float));
  float *B = (float *)aligned_alloc(16 * sizeof(float), K * sizeof(float));
  float *C = (float *)aligned_alloc(16 * sizeof(float), M * sizeof(float));
  float *C_ground_truth = (float *)aligned_alloc(16 * sizeof(float), M * sizeof(float));
  fill_random(A, M * K);
  fill_random(B, K);
  fill_random(C, M);
  fill_random(C_ground_truth, M);

  LIKWID_MARKER_START("Compute");
  start = clock();
  for (int i = 0; i < repeats; i++) {
    function(A, B, C, M, K);
  }
  end = clock();
  LIKWID_MARKER_STOP("Compute");

  // get ground truth 
  matvec_1(A, B, C_ground_truth, M, K);
  bool correctness = is_same(C, C_ground_truth, M);

  free(A);
  free(B);
  free(C);
  free(C_ground_truth);

  double cpu_time_ms = (double)(end - start) / CLOCKS_PER_SEC * 1e3;
  long long flop = M * K * 2L * repeats;
  double flops = (double)flop / (cpu_time_ms / 1e3);

  experiment_result_t results;
  results.ms = cpu_time_ms;
  results.flops = flops;
  results.flop = flop;
  results.clock_cycles = end - start;
  results.correctness = correctness;
  return results;
}

int main(int argc, char **argv) {
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;

  if (argc <= 2) {
    printf("usage: ./run function_num function_param\n");
    return 1;
  }

  void (*peak_flops_funcs[32])(float *, float *, float *, int, int) =
      FTABLE_0_31(peak_flops);
  void (*matvec_2_funcs[32])(float *, float *, float *, int, int) =
      FTABLE_0_31(matvec_2);
  void (*matvec_3_funcs[32])(float *, float *, float *, int, int) =
      FTABLE_0_31(matvec_3);

  int function_num = atoi(argv[1]);
  int function_parameter = atoi(argv[2]);

  void (*function)(float *, float *, float *, int, int);
  switch (function_num) {
  case 0:
    // get peak flops
    function = peak_flops_funcs[function_parameter];
    break;
  case 1:
    function = &matvec_1;
    break;
  case 2:
    function = matvec_2_funcs[function_parameter];
    break;
  case 3:
    function = matvec_3_funcs[function_parameter];
    break;
  default:
    printf("Unrecognized function number %d", function_num);
    return 1;
  }

  experiment_result_t result =
      measure_condition(1024 * 8, 1024 * 8, REPEATS, function);
  printf("%d,", function_num);
  printf("%d,", function_parameter);
  printf("%.4f,", result.ms);
  printf("%lld,", result.flop);
  printf("%.4e,", result.flops);
  printf("%d\n", result.correctness);
  LIKWID_MARKER_CLOSE;
  return 0;
}