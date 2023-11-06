#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>

#ifndef __MATRIX_H__
#define __MATRIX_H__

namespace support {

template <typename T>
class Matrix {
private:
    T* _data;
    uint32_t _width, _height;

public:
    // Constructor
    Matrix(uint32_t width, uint32_t height) : _width(width), _height(height) {
        _data = new T[_width * _height];
    }

    // Destructor
    ~Matrix() {
        if (_data != nullptr) {
            delete[] _data;
        }
    }

    // Move constructor
    Matrix(Matrix&& other) : _width(other._width), _height(other._height) {
        _data = other._data;
        other._data = nullptr;      
    }

    Matrix<T>&& copy() {
        Matrix<T> result(_width, _height);
        std::memcpy(result._data, _data, sizeof(T) * _width * _height);
        return result;
    }

    // Move assignment constructor
    Matrix<T>& operator=(Matrix&& other) {
        if (_data != nullptr) {
            delete[] _data;
        }
        
        _data = other._data;
        _width = other._width;
        _height = other._height;

        other._data = nullptr;    
        return *this;
    }

    inline T& access(uint32_t r, uint32_t c) {
        // NOTE: does not check indices are valid, be careful!
        uint32_t flat_index = r * _width + c;
        return _data[flat_index];         
    }

    inline T& a(uint32_t r, uint32_t c) {
        return access(r, c);
    }

    inline const T& read(uint32_t r, uint32_t c) const{
        // NOTE: does not check indices are valid, be careful!
        uint32_t flat_index = r * _width + c;
        return _data[flat_index];    
    }

    inline const T& r(uint32_t r, uint32_t c) const {
        return read(r, c);
    }

    uint32_t get_height() const {
        return _height;
    };

    uint32_t get_width() const {
        return _width;
    }
};

template<typename T>
void print(std::ostream& os, const Matrix<T>& arr) {
    static const uint32_t MAX_ROWS = 10;
    static const uint32_t MAX_COLS = 10;

    bool fit_rows = arr.get_height() <= MAX_ROWS;
    bool fit_cols = arr.get_width() <= MAX_COLS;

    for (int r = 0; r < std::min(MAX_ROWS, arr.get_height()); r++) {
        for (int c = 0; c < std::min(MAX_COLS, arr.get_width()); c++) {
            os << arr.r(r, c) << " ";
        }

        os << (fit_cols ? "" : "...") << std::endl;
    }

    os << (fit_rows ? "" : "...") << std::endl;;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& arr) {
    print(os, arr);
    return os;
}

} // namespace support

#endif