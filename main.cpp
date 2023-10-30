#include "matrix.h"
#include <iostream>

using support::Matrix;

int main() {
    std::cout << "Hello world!" << std::endl;

    Matrix<float> big_matrix(100, 100);
    std::cout << big_matrix << std::endl;

    Matrix<float> small_matrix(5, 5);
    std::cout << small_matrix << std::endl;

    return 0;
}