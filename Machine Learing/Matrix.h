#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>

class Matrix {
public:
    // Constructors
    Matrix(unsigned rows, unsigned cols);
    Matrix(unsigned size);  // Constructor for identity matrix

    // Basic operations like addition, subtraction, multiplication
    Matrix operator+(const Matrix& rhs);
    Matrix operator-(const Matrix& rhs);
    Matrix operator*(const Matrix& rhs) const;

    // Scalar operations
    Matrix operator*(double scalar);
    Matrix operator/(double scalar);

    // Utility functions
    Matrix transpose() const;
    void print() const;

    // New features
    double determinant() const;
    double frobeniusNorm() const;
    bool operator==(const Matrix& rhs) const;
    Matrix inverse() const; 

    // LU Decomposition
    void LU_decompose(Matrix& L, Matrix& U) const;

    // Power Iteration Method for largest eigenvalue and eigenvector
    double powerIterationEigenvalue() const;
    Matrix powerIterationEigenvector() const;
    Matrix normalize() const;
    
     // Public getters for private members
     double at(unsigned i, unsigned j) const {
        if (i < rows && j < cols) {
            return data[i][j];
        } else {
        // Handle the error, e.g., by throwing an exception
            throw std::out_of_range("Index out of range");
        }
    }
    unsigned getRows() const { return rows; }
    unsigned getCols() const { return cols; }

private:
    std::vector<std::vector<double>> data;
    unsigned rows, cols;
};
