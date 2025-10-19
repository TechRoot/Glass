#include "qr.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    using lin::Matrix;
    // Matriz de ejemplo (3x3) para descomposición QR
    Matrix A{{12.0, -51.0, 4.0},
             {6.0, 167.0, -68.0},
             {-4.0, 24.0, -41.0}};
    Matrix Q, R;
    lin::qr_decompose(A, Q, R);
    // Comprobar que Q * R se aproxima a A
    Matrix QR = lin::matmul(Q, R);
    for (std::size_t i = 0; i < A.size(); ++i) {
        for (std::size_t j = 0; j < A[0].size(); ++j) {
            double diff = std::fabs(QR[i][j] - A[i][j]);
            assert(diff < 1e-6);
        }
    }
    // Comprobar que Q es ortonormal: Q^T * Q = I
    Matrix Qt = lin::transpose(Q);
    Matrix QtQ = lin::matmul(Qt, Q);
    for (std::size_t i = 0; i < QtQ.size(); ++i) {
        for (std::size_t j = 0; j < QtQ[0].size(); ++j) {
            double expected = (i == j ? 1.0 : 0.0);
            double diff = std::fabs(QtQ[i][j] - expected);
            assert(diff < 1e-6);
        }
    }
    // Resolver sistema A x = b usando QR (2x2)
    Matrix A2{{2.0, 1.0}, {1.0, 3.0}};
    std::vector<double> b{3.0, 5.0};
    Matrix Q2, R2;
    lin::qr_decompose(A2, Q2, R2);
    std::vector<double> x = lin::solve_qr(Q2, R2, b);
    // Solución exacta: x = [1, 1]
    assert(std::fabs(x[0] - 1.0) < 1e-6);
    assert(std::fabs(x[1] - 1.0) < 1e-6);
    std::cout << "All QR tests passed\n";
    return 0;
}