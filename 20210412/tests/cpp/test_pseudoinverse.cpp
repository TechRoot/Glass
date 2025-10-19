#include "pseudoinverse.hpp"
#include <cmath>
#include <iostream>
#include <vector>
#include <cassert>

// Función auxiliar para multiplicar matrices
static lin::Matrix matmul(const lin::Matrix &A, const lin::Matrix &B) {
    size_t m = A.size();
    size_t n = B[0].size();
    size_t inner = B.size();
    lin::Matrix C(m, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < m; ++i) {
        for (size_t k = 0; k < inner; ++k) {
            double a = A[i][k];
            if (a != 0.0) {
                for (size_t j = 0; j < n; ++j) {
                    C[i][j] += a * B[k][j];
                }
            }
        }
    }
    return C;
}

static bool matrices_approx_equal(const lin::Matrix &A, const lin::Matrix &B, double eps = 1e-6) {
    if (A.size() != B.size() || A.empty() || B.empty() || A[0].size() != B[0].size()) {
        return false;
    }
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            if (std::fabs(A[i][j] - B[i][j]) > eps) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    // Test 1: inversa de matriz cuadrada (2x2)
    lin::Matrix A{{4.0, 7.0}, {2.0, 6.0}};
    lin::Matrix invA = lin::pseudoinverse(A);
    // La pseudoinversa debe coincidir con la inversa exacta
    lin::Matrix expected_inv{{0.6, -0.7}, {-0.2, 0.4}};
    assert(matrices_approx_equal(invA, expected_inv, 1e-6));
    // Verificar A * A⁺ ≈ I
    lin::Matrix I = matmul(A, invA);
    lin::Matrix ident{{1.0, 0.0}, {0.0, 1.0}};
    assert(matrices_approx_equal(I, ident, 1e-6));

    // Test 2: matriz rectangular (3x2)
    lin::Matrix B{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    lin::Matrix Bplus = lin::pseudoinverse(B);
    // verificar propiedades de la pseudoinversa: B B⁺ B = B
    lin::Matrix B1 = matmul(matmul(B, Bplus), B);
    assert(matrices_approx_equal(B1, B, 1e-5));
    // B⁺ B B⁺ = B⁺
    lin::Matrix B2 = matmul(matmul(Bplus, B), Bplus);
    assert(matrices_approx_equal(B2, Bplus, 1e-5));
    // Número de condición debe ser finito
    double cond = lin::condition_number(B);
    assert(cond > 0.0 && cond < 1e4);
    std::cout << "Todas las pruebas de pseudoinversa se han superado satisfactoriamente.\n";
    return 0;
}