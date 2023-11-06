#include "matrix.h"
#include "naive_matmul.h"
#include <iostream>
#include <random>


using support::Matrix;

void randomInitFloatMatrix(Matrix<float>& mat) {
  std::default_random_engine generator(0);
  std::uniform_real_distribution<float> distribution(0.0,1.0);

    for (int r = 0; r < mat.get_height(); r++) {
        for (int c = 0; c < mat.get_width(); c++) {
            mat.a(r, c) = distribution(generator);
        }
    }
}

int main() {
    std::cout << "Hello world!" << std::endl;
    Matrix<float> matA(5, 5);
    Matrix<float> matB(5, 5);
    Matrix<float> matC(5, 5);

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

    return 0;
}