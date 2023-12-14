// Matrix.cpp
#include "Matrix.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <limits>

int main() {
    Matrix myMatrix(3, 3);  // Create a 3x3 matrix initialized to 0
    std::cout << "Matrix:" << std::endl;
    myMatrix.print();  // Print the matrix to the console

    // Since the matrix is all zeros, the determinant should be 0
    std::cout << "Determinant: " << myMatrix.determinant() << std::endl;

    // Frobenius norm should also be 0 since all elements are 0
    std::cout << "Frobenius Norm: " << myMatrix.frobeniusNorm() << std::endl;

    // Create an identity matrix of size 3 and print it
    Matrix identityMatrix(3);
    std::cout << "Identity Matrix:" << std::endl;
    identityMatrix.print();

    // Check if myMatrix is equal to identityMatrix (should be false)
    if (myMatrix == identityMatrix) {
        std::cout << "myMatrix and identityMatrix are equal." << std::endl;
    } else {
        std::cout << "myMatrix and identityMatrix are not equal." << std::endl;
    }

    return 0;
}


Matrix::Matrix(unsigned size) : data(size, std::vector<double>(size, 0.0)), rows(size), cols(size) {
    for (unsigned i = 0; i < size; ++i) {
        data[i][i] = 1.0;
    }
}

Matrix::Matrix(unsigned rows, unsigned cols) : data(rows, std::vector<double>(cols, 0.0)), rows(rows), cols(cols) {}

Matrix Matrix::transpose() const {
    Matrix transposed(cols, rows);
    for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
            transposed.data[j][i] = data[i][j];
        }
    }
    return transposed;
}

Matrix Matrix::operator+(const Matrix& rhs) {
    Matrix result(rows, cols);
    for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
            result.data[i][j] = this->data[i][j] + rhs.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& rhs) {
    if (rows != rhs.rows || cols != rhs.cols) {
        throw std::invalid_argument("Matrices must have the same dimensions for subtraction.");
    }
    Matrix result(rows, cols);
    for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
            result.data[i][j] = this->data[i][j] - rhs.data[i][j];
        }
    }
    return result;
}
Matrix Matrix::operator*(const Matrix& rhs) const {
    if (cols != rhs.rows) {
        throw std::invalid_argument("Matrix multiplication dimension mismatch.");
    }
    Matrix product(rows, rhs.cols);
    for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < rhs.cols; ++j) {
            for (unsigned k = 0; k < cols; ++k) {
                product.data[i][j] += data[i][k] * rhs.data[k][j];
            }
        }
    }
    return product;
}

Matrix Matrix::operator*(double scalar) {
    Matrix result(rows, cols);
    for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
            result.data[i][j] = this->data[i][j] * scalar;
        }
    }
    return result;
}

Matrix Matrix::operator/(double scalar) {
    if (scalar == 0) {
        throw std::invalid_argument("Division by zero.");
    }
    Matrix result(rows, cols);
    for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
            result.data[i][j] = this->data[i][j] / scalar;
        }
    }
    return result;
}


void Matrix::print() const {
    for (const auto& row : data) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

// Determinant
double Matrix::determinant() const {
    if (rows != cols) {
        throw std::invalid_argument("Determinant is only defined for square matrices.");
    }
    if (rows == 1) {
        return data[0][0];
    }
    double det = 0.0;
    int sign = 1;
    for (unsigned i = 0; i < rows; ++i) {
        Matrix subMatrix(rows - 1, cols - 1);
        for (unsigned m = 1; m < rows; ++m) {
            for (unsigned n = 0, col = 0; n < cols; ++n) {
                if (n != i) {
                    subMatrix.data[m - 1][col] = data[m][n];
                    col++;
                }
            }
        }
        det += sign * data[0][i] * subMatrix.determinant();
        sign = -sign;
    }
    return det;
}

// Frobenius Norm
double Matrix::frobeniusNorm() const {
    double sum = 0.0;
    for (auto& row : data) {
        for (double val : row) {
            sum += val * val;
        }
    }
    return std::sqrt(sum);
}

// Equality Operator
bool Matrix::operator==(const Matrix& rhs) const {
    if (rows != rhs.rows || cols != rhs.cols) {
        return false;
    }
    for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
            if (data[i][j] != rhs.data[i][j]) {
                return false;
            }
        }
    }
    return true;
}

Matrix Matrix::inverse() const {
    if (rows != cols) {
        throw std::invalid_argument("Inverse is only defined for square matrices.");
    }

    // Create an augmented matrix with the identity matrix
    Matrix augmented(rows, cols * 2);
    for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
            augmented.data[i][j] = data[i][j];
            augmented.data[i][j + cols] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Perform Gauss-Jordan elimination
    for (unsigned i = 0; i < rows; ++i) {
        // Make sure the pivot element is non-zero
        if (augmented.data[i][i] == 0.0) {
            throw std::invalid_argument("Matrix is singular and cannot be inverted.");
        }

        // Scale the pivot row
        double scale = 1.0 / augmented.data[i][i];
        for (unsigned j = 0; j < augmented.cols; ++j) {
            augmented.data[i][j] *= scale;
        }

        // Make all other column elements in this row 0
        for (unsigned k = 0; k < rows; ++k) {
            if (k != i) {
                double scale_factor = augmented.data[k][i];
                for (unsigned j = 0; j < augmented.cols; ++j) {
                    augmented.data[k][j] -= scale_factor * augmented.data[i][j];
                }
            }
        }
    }

    // Extract the inverse matrix from the augmented matrix
    Matrix inverse(rows, cols);
    for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
            inverse.data[i][j] = augmented.data[i][j + cols];
        }
    }

    return inverse;
}

void Matrix::LU_decompose(Matrix& L, Matrix& U) const {
    if (rows != cols) {
        throw std::invalid_argument("LU decomposition requires a square matrix.");
    }

    // Initialize L and U
    L = Matrix(rows, cols);
    U = Matrix(*this); // Copy the original matrix into U

    for (unsigned i = 0; i < rows; ++i) {
        L.data[i][i] = 1.0; // Set the diagonal of L to 1
    }

    for (unsigned i = 0; i < rows - 1; ++i) {
        // Partial pivoting
        unsigned maxIndex = i;
        double maxValue = std::abs(U.data[i][i]);
        for (unsigned k = i + 1; k < rows; ++k) {
            if (std::abs(U.data[k][i]) > maxValue) {
                maxValue = std::abs(U.data[k][i]);
                maxIndex = k;
            }
        }
        if (maxIndex != i) {
            // Swap rows in U
            std::swap(U.data[i], U.data[maxIndex]);
            // Swap rows in L (except the diagonal element)
            for (unsigned j = 0; j < i; ++j) {
                std::swap(L.data[i][j], L.data[maxIndex][j]);
            }
        }

        // Continue with LU decomposition
        for (unsigned k = i + 1; k < rows; ++k) {
            L.data[k][i] = U.data[k][i] / U.data[i][i];
            for (unsigned j = i; j < rows; ++j) {
                U.data[k][j] -= L.data[k][i] * U.data[i][j];
            }
        }
    }
}

Matrix Matrix::normalize() const {
    double norm = 0.0;
    for (unsigned i = 0; i < this->getRows(); ++i) {
        double val = this->at(i, 0);
        norm += val * val;
    }
    norm = std::sqrt(norm);

    Matrix result(this->getRows(), 1);
    for (unsigned i = 0; i < this->getRows(); ++i) {
        result.data[i][0] = this->at(i, 0) / norm;
    }
    return result;
}


double Matrix::powerIterationEigenvalue() const {
    // Initial random vector
    Matrix b(rows, 1);
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (unsigned i = 0; i < rows; ++i) {
        b.data[i][0] = distribution(generator);
    }

    Matrix b_next = *this * b.normalize();

    double eigenvalue;
    while (true) {
        b_next = (*this * b).normalize();
        Matrix multiplied = *this * b_next;

        eigenvalue = multiplied.at(0, 0); // Assuming multiplied is a 1x1 matrix

        if ((b_next - b).normalize().frobeniusNorm() < 1e-6) {
            break;
        }
        b = b_next;
    }
    return eigenvalue;
}

Matrix Matrix::powerIterationEigenvector() const {
    Matrix b(rows, 1);
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (unsigned i = 0; i < rows; ++i) {
        b.data[i][0] = distribution(generator);
    }

    Matrix b_next = *this * b;
    b = b.normalize();

    // Iterate until convergence
    while (true) {
        Matrix b_next = *this * b; // Multiply with the matrix
        b_next = b_next.normalize(); // Normalize the result

        // Convergence check
        if ((b_next - b).frobeniusNorm() < std::numeric_limits<double>::epsilon()) {
            return b_next;
        }

        b = b_next; // Prepare for next iteration
    }
}
// TODO: Implement other matrix operations
