#pragma once
#include <cstring>

// Simple replacement for std::vector with better NUMA awareness
namespace my
{

template<typename T>
class vector
{

private:

    // Internal storage
    T *_data = nullptr;

public:

    // dimension
    int n;

    // Default constructor
    vector() = default;
    // Allocate at the time of construction
    vector(int n) : n(n) {
        _data = new T [n];
    };

   void resize(int n_in) {
        n = n_in;
        delete[] _data;
        _data = new T [n];
    };


    // standard [i] syntax for setting elements
    T& operator[](int i) {
        return _data[i];
    }

    // standard [i] syntax for getting elements
    const T& operator[](int i) const {
        return _data[i];
    }

// Rule of five when we manage memory ourselves    
    // Copy constructor
    vector(const vector& other) {
      n = other.n;
      _data = new T [n];
      std::memcpy(_data, other._data, n*sizeof(T));
    }

    // Copy assignment
    vector& operator= (const vector& other) {
      auto tmp = other;
      std::swap(n, tmp.n);
      std::swap(_data, tmp._data);
      return *this;
    }

    // Move constructor
    vector(vector&& other) {
      n = other.n;
      _data = other._data;
      other._data = nullptr;
    }
    // Move assignment
    vector& operator= (vector&& other) {
      n = other.n;
      _data = other._data;
      other._data = nullptr;
      return *this;
    }

    // Destructor
    ~vector() {
       delete[] _data;
     }

    // provide possibility to get raw pointer for data at index (i) (needed for MPI)
    T* data(int i=0) {
        return _data + i;
    }
};
};
