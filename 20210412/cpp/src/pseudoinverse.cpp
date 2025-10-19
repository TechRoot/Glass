#include "pseudoinverse.hpp"
#include <stdexcept>
#include <cmath>

namespace lin {

static Matrix multiply(const Matrix &A, const Matrix &B) {
    size_t m = A.size();
    size_t n = B[0].size();
    size_t inner = B.size();
    Matrix C(m, std::vector<double>(n, 0.0));
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

static Matrix transpose(const Matrix &A) {
    size_t m = A.size();
    size_t n = A[0].size();
    Matrix T(n, std::vector<double>(m));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

Matrix inverse(const Matrix &A) {
    size_t n = A.size();
    if (A.empty() || A[0].size() != n) {
        throw std::runtime_error("La matriz debe ser cuadrada para invertirla");
    }
    // matriz aumentada
    Matrix aug(n, std::vector<double>(2 * n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            aug[i][j] = A[i][j];
        }
        for (size_t j = 0; j < n; ++j) {
            aug[i][n + j] = (i == j ? 1.0 : 0.0);
        }
    }
    // eliminación de Gauss-Jordan
    for (size_t i = 0; i < n; ++i) {
        // seleccionar pivote
        size_t pivot = i;
        for (size_t r = i; r < n; ++r) {
            if (std::fabs(aug[r][i]) > std::fabs(aug[pivot][i])) {
                pivot = r;
            }
        }
        if (std::fabs(aug[pivot][i]) < 1e-12) {
            throw std::runtime_error("Matriz singular, no se puede invertir");
        }
        if (pivot != i) {
            std::swap(aug[pivot], aug[i]);
        }
        double diag = aug[i][i];
        // normalizar la fila
        for (size_t j = 0; j < 2 * n; ++j) {
            aug[i][j] /= diag;
        }
        // eliminar otras filas
        for (size_t r = 0; r < n; ++r) {
            if (r == i) continue;
            double factor = aug[r][i];
            if (factor != 0.0) {
                for (size_t j = 0; j < 2 * n; ++j) {
                    aug[r][j] -= factor * aug[i][j];
                }
            }
        }
    }
    // extraer inversa
    Matrix inv(n, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            inv[i][j] = aug[i][n + j];
        }
    }
    return inv;
}

Matrix pseudoinverse(const Matrix &A) {
    size_t m = A.size();
    size_t n = A.empty() ? 0 : A[0].size();
    if (m == 0 || n == 0) {
        return Matrix();
    }
    Matrix AT = transpose(A);
    if (m >= n) {
        // usar A+ = (A^T A)^-1 A^T
        Matrix ATA = multiply(AT, A);
        Matrix ATA_inv = inverse(ATA);
        return multiply(ATA_inv, AT);
    } else {
        // usar A+ = A^T (A A^T)^-1
        Matrix AAT = multiply(A, AT);
        Matrix AAT_inv = inverse(AAT);
        return multiply(AT, AAT_inv);
    }
}

static double frobenius_norm(const Matrix &A) {
    double sum = 0.0;
    for (const auto &row : A) {
        for (double v : row) {
            sum += v * v;
        }
    }
    return std::sqrt(sum);
}

double condition_number(const Matrix &A) {
    Matrix pinv = pseudoinverse(A);
    double normA = frobenius_norm(A);
    double normInv = frobenius_norm(pinv);
    return normA * normInv;
}

} // namespace lin