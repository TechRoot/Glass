/*
 * Implementación básica de la factorización QR mediante reflexiones de
 * Householder y utilidades asociadas para resolver sistemas lineales.
 *
 * Las matrices se representan como `std::vector<std::vector<double>>` de
 * tamaño m×n.  Las funciones proporcionan operaciones elementales para
 * actualizar las matrices y calcular el resultado del producto Q·R.
 *
 * Aunque existen bibliotecas especializadas como Eigen o LAPACK, esta
 * implementación es educativa y suficiente para matrices pequeñas.
 */

#ifndef QR_DECOMP_HPP
#define QR_DECOMP_HPP

#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace lin {

using Matrix = std::vector<std::vector<double>>;

/**
 * Crea una matriz identidad de tamaño n×n.
 */
inline Matrix identity(std::size_t n) {
    Matrix I(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) I[i][i] = 1.0;
    return I;
}

/**
 * Transpone una matriz.
 */
inline Matrix transpose(const Matrix &A) {
    std::size_t m = A.size();
    std::size_t n = A[0].size();
    Matrix B(n, std::vector<double>(m));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            B[j][i] = A[i][j];
        }
    }
    return B;
}

/**
 * Multiplica dos matrices A (m×n) y B (n×p).
 */
inline Matrix matmul(const Matrix &A, const Matrix &B) {
    std::size_t m = A.size();
    std::size_t n = A[0].size();
    std::size_t p = B[0].size();
    Matrix C(m, std::vector<double>(p, 0.0));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            double a = A[i][k];
            for (std::size_t j = 0; j < p; ++j) {
                C[i][j] += a * B[k][j];
            }
        }
    }
    return C;
}

/**
 * Calcula la norma 2 de un vector a partir de sus componentes en filas
 * consecutivas de una columna de una matriz.
 */
inline double column_norm(const Matrix &A, std::size_t col, std::size_t start) {
    double sum = 0.0;
    for (std::size_t i = start; i < A.size(); ++i) {
        sum += A[i][col] * A[i][col];
    }
    return std::sqrt(sum);
}

/**
 * Aplica la reflexión de Householder a la submatriz a partir de la fila
 * `start` y la columna `col`.  Modifica A y acumula la transformación en Q.
 */
inline void householder_reflect(Matrix &A, Matrix &Q, std::size_t start, std::size_t col) {
    std::size_t m = A.size();
    // Construir vector x a partir de la columna actual
    std::vector<double> x(m - start);
    for (std::size_t i = start; i < m; ++i) {
        x[i - start] = A[i][col];
    }
    double norm_x = 0.0;
    for (double v : x) norm_x += v * v;
    norm_x = std::sqrt(norm_x);
    if (norm_x == 0.0) return;
    // Construir vector v = x + sign(x0) * ||x|| * e1
    std::vector<double> v = x;
    v[0] += (x[0] >= 0 ? norm_x : -norm_x);
    // Normalizar v
    double norm_v = 0.0;
    for (double t : v) norm_v += t * t;
    norm_v = std::sqrt(norm_v);
    if (norm_v == 0.0) return;
    for (double &t : v) t /= norm_v;
    // Construir H = I - 2 v v^T y aplicarlo a A[start:m, col:n]
    std::size_t n = A[0].size();
    // Aplicar H a la submatriz A: para cada columna j calculamos la proyección
    // de la columna sobre el vector v y restamos 2*proy*v en la región afectada.
    for (std::size_t j = col; j < n; ++j) {
        double proj = 0.0;
        for (std::size_t k = 0; k < v.size(); ++k) {
            proj += v[k] * A[start + k][j];
        }
        proj *= 2.0;
        for (std::size_t k = 0; k < v.size(); ++k) {
            A[start + k][j] -= proj * v[k];
        }
    }
    // Aplicar H acumulativo a Q (Q = Q H^T) para construcción explícita de Q
    std::size_t qrows = Q.size();
    for (std::size_t j = 0; j < qrows; ++j) {
        double proj = 0.0;
        for (std::size_t k = 0; k < v.size(); ++k) {
            proj += v[k] * Q[j][start + k];
        }
        proj *= 2.0;
        for (std::size_t k = 0; k < v.size(); ++k) {
            Q[j][start + k] -= proj * v[k];
        }
    }
}

/**
 * Calcula la factorización QR de la matriz A.
 *
 * @param A Matriz de tamaño m×n (se copia internamente).
 * @param Q Matriz ortogonal de tamaño m×m (devuelta por referencia).
 * @param R Matriz triangular superior de tamaño m×n (devuelta por referencia).
 */
inline void qr_decompose(const Matrix &A_in, Matrix &Q, Matrix &R) {
    std::size_t m = A_in.size();
    std::size_t n = A_in[0].size();
    // Inicializar R como copia de A_in
    R = A_in;
    // Inicializar Q como identidad
    Q = identity(m);
    std::size_t min_dim = m < n ? m : n;
    for (std::size_t col = 0; col < min_dim; ++col) {
        householder_reflect(R, Q, col, col);
    }
    // Ajustar R a triangular superior exactamente (anula pequeños valores numéricos)
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < i && j < n; ++j) {
            R[i][j] = 0.0;
        }
    }
    // Q es ortonormal por filas; transponer para representar Q tradicional (columnas ortonormales)
    Q = transpose(Q);
}

/**
 * Resuelve el sistema A x = b usando la factorización QR previamente calculada.
 *
 * @param Q Matriz ortogonal m×m.
 * @param R Matriz triangular superior m×n.
 * @param b Vector de observaciones de tamaño m.
 * @return Vector solución de tamaño n.
 */
inline std::vector<double> solve_qr(const Matrix &Q, const Matrix &R, const std::vector<double> &b) {
    std::size_t m = Q.size();
    std::size_t n = R[0].size();
    if (b.size() != m) throw std::invalid_argument("Dimensiones incompatibles");
    // Calcular y = Q^T b
    std::vector<double> y(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            y[i] += Q[j][i] * b[j];
        }
    }
    // Resolver R x = y por sustitución hacia atrás
    std::vector<double> x(n, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        double sum = 0.0;
        for (std::size_t j = i + 1; j < n; ++j) {
            sum += R[i][j] * x[j];
        }
        double val = y[i] - sum;
        double diag = R[i][i];
        if (std::abs(diag) < 1e-12) throw std::runtime_error("Matriz singular en solve_qr");
        x[i] = val / diag;
    }
    return x;
}

/**
 * Calcula la norma 2 de un vector.
 */
inline double norm2(const std::vector<double> &v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

}  // namespace lin

#endif  // QR_DECOMP_HPP