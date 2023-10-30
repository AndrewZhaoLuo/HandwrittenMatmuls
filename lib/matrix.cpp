#include <stdint.h>
#include <stdlib.h>
#include "matrix.h"

namespace support {

    // constructor
    template<typename T>
    Matrix<T>::Matrix(uint32_t width, uint32_t height) : _width(width), _height(height) {
        _data = new T[_width * _height];
    }

    // destructor
    template<typename T>
    Matrix<T>::~Matrix() {
        if (_data != nullptr) {
            delete[] _data;
        }
    }

    // move constructor
    template<typename T>
    Matrix<T>::Matrix(Matrix&& other) : _width(other._width), _height(other._height) {
        _data = other._data;
        other._data = nullptr;      
    }

    // move assignment constructor
    template<typename T>
    Matrix<T>& Matrix<T>::operator=(Matrix&& other) {
        _data = other._data;
        _width = other._width;
        _height = other._height;

        other._data = nullptr;    
        return *this;
    }

    template<typename T>
    T& Matrix<T>::access(uint32_t r, uint32_t c) {
        // NOTE: does not check indices are valid, be careful!
        uint32_t flat_index = r * _width + c;
        return _data[flat_index];         
    }

} // namespace support