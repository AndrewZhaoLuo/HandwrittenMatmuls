#include <stdint.h>
#include <stdlib.h>
#include <iostream>

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

    inline T& access(uint32_t r, uint32_t c); 
    inline T& a(uint32_t r, uint32_t c); 

    uint32_t get_height() const;
    uint32_t get_width() const;
};

template <typename T>
void print(std::ostream&, const Matrix<T>&);

} // namespace support

#endif