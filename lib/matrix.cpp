#include <stdint.h>
#include <stdlib.h>
#include <iostream>
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
    inline T& Matrix<T>::access(uint32_t r, uint32_t c) {
        // NOTE: does not check indices are valid, be careful!
        uint32_t flat_index = r * _width + c;
        return _data[flat_index];         
    }

    template<typename T>
    inline T& Matrix<T>::a(uint32_t r, uint32_t c) {
        return access(r, c);
    }

    template<typename T>
    uint32_t Matrix<T>::get_height() const {
        return _height;
    };

    template<typename T>
    uint32_t Matrix<T>::get_width() const {
        return _width;
    }

    template<typename T>
    void print(std::ostream& os, const Matrix<T>& arr) {
        static const uint32_t MAX_ROWS = 10;
        static const uint32_t MAX_COLS = 10;

        bool fit_rows = arr.get_height() <= MAX_ROWS;
        bool fit_cols = arr.get_width() <= MAX_COLS;

        for (int r = 0; r < std::min(MAX_ROWS, arr.get_height()); r++) {
            for (int c = 0; c < std::min(MAX_COLS, arr.get_width()); c++) {
                os << arr.a(r, c) << " ";
            }

            os << (fit_cols ? "" : "...") << std::endl;
        }

        os << (fit_rows ? "" : "...") << std::endl;;
    }

} // namespace support