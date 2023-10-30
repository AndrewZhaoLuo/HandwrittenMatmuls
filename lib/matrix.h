#include <stdint.h>
#include <stdlib.h>

#ifndef __MATRIX_H__
#define __MATRIX_H__

namespace support {

template <typename T>
class Matrix {
private:
    T* _data;
    uint32_t _width, _height;

public:

    Matrix(uint32_t, uint32_t);  // constructor
    ~Matrix();                   // destructor
    Matrix(Matrix&&);            // move constructor
    Matrix& operator=(Matrix&&); // move assignment constructor

    T& access(uint32_t r, uint32_t c); 
};

} // namespace support

#endif