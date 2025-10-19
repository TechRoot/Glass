"""Módulo de ajuste por mínimos cuadrados.

Este módulo proporciona funciones para resolver problemas de regresión
lineal mediante distintos enfoques: ecuaciones normales, descomposición
en valores singulares (SVD) y factorización QR.  También incluye una
versión regularizada (ridge) basada en las ecuaciones normales y una
función para calcular el error cuadrático medio del ajuste.

Las matrices de diseño `A` pueden ser de tamaño m×n con m ≥ n.  Se
retorna siempre el vector de coeficientes que minimiza ||A·x − b||₂.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


def normal_equations(A: Sequence[Sequence[float]], b: Sequence[float], lam: float = 0.0) -> Tuple[np.ndarray, float]:
    """Resuelve el problema de mínimos cuadrados mediante ecuaciones normales.

    Calcula x = (A^T A + λI)^{-1} A^T b. El parámetro λ permite añadir
    regularización de Tikhonov (ridge) para mejorar la estabilidad en
    matrices mal condicionadas. Devuelve la solución y el número de
    condición de A^T A + λI.

    Parámetros
    ----------
    A : secuencia de secuencias
        Matriz de diseño m×n.
    b : secuencia
        Vector de observaciones de longitud m.
    lam : float, opcional
        Parámetro de regularización (λ). Por defecto 0 para el caso clásico.

    Retorna
    -------
    Tuple[np.ndarray, float]
        Un par (x, cond), donde x son los coeficientes y cond es el número
        de condición de la matriz normalizada.
    """
    A_mat = np.asarray(A, dtype=float)
    b_vec = np.asarray(b, dtype=float)
    # Formar A^T A y A^T b
    ATA = A_mat.T @ A_mat
    if lam != 0.0:
        ATA = ATA + lam * np.eye(ATA.shape[0])
    ATb = A_mat.T @ b_vec
    # Resolver el sistema lineal
    x = np.linalg.solve(ATA, ATb)
    # Número de condición
    cond = np.linalg.cond(ATA)
    return x, float(cond)


def svd_least_squares(A: Sequence[Sequence[float]], b: Sequence[float]) -> Tuple[np.ndarray, float]:
    """Resuelve el problema de mínimos cuadrados usando SVD.

    Se utiliza `numpy.linalg.lstsq`, que emplea SVD internamente para
    obtener la solución de norma mínima.  Devuelve la solución y el
    número de condición de A.

    Parámetros
    ----------
    A : secuencia de secuencias
        Matriz de diseño m×n.
    b : secuencia
        Vector de observaciones de longitud m.

    Retorna
    -------
    Tuple[np.ndarray, float]
        Un par (x, cond), donde x son los coeficientes y cond es el número
        de condición de A.
    """
    A_mat = np.asarray(A, dtype=float)
    b_vec = np.asarray(b, dtype=float)
    x, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    # El número de condición se calcula como ratio entre singular values máximo y mínimo
    if s.size > 0:
        cond = s.max() / s.min() if s.min() != 0 else float('inf')
    else:
        cond = float('inf')
    return x, float(cond)


def qr_least_squares(A: Sequence[Sequence[float]], b: Sequence[float]) -> Tuple[np.ndarray, float]:
    """Resuelve un problema de mínimos cuadrados usando factorización QR.

    Se emplea la factorización QR de NumPy (`numpy.linalg.qr`) para
    resolver A·x = b de forma estable.  Devuelve la solución y el número
    de condición de R.

    Parámetros
    ----------
    A : secuencia de secuencias
        Matriz de diseño m×n.
    b : secuencia
        Vector de observaciones.

    Retorna
    -------
    Tuple[np.ndarray, float]
        Par (x, cond), donde x son los coeficientes y cond es el número de
        condición de R.
    """
    A_mat = np.asarray(A, dtype=float)
    b_vec = np.asarray(b, dtype=float)
    # Factorizar A = Q R
    Q, R = np.linalg.qr(A_mat)
    # Resolver R x = Q^T b
    Qt_b = Q.T @ b_vec
    x = np.linalg.solve(R, Qt_b)
    # Número de condición de R
    cond = np.linalg.cond(R)
    return x, float(cond)


def ridge_regression(A: Sequence[Sequence[float]], b: Sequence[float], alpha: float) -> Tuple[np.ndarray, float]:
    """Realiza regresión ridge resolviendo (A^T A + αI)x = A^T b.

    Esta función es un alias de `normal_equations` con el parámetro
    lam=alpha.  Retorna los coeficientes y el número de condición de la
    matriz regularizada.

    Parámetros
    ----------
    A : secuencia de secuencias
        Matriz de diseño m×n.
    b : secuencia
        Vector de observaciones.
    alpha : float
        Parámetro de regularización (α > 0).

    Retorna
    -------
    Tuple[np.ndarray, float]
        Coeficientes de la regresión y el número de condición.
    """
    return normal_equations(A, b, lam=alpha)


def mean_squared_error(A: Sequence[Sequence[float]], x: Sequence[float], b: Sequence[float]) -> float:
    """Calcula el error cuadrático medio de un ajuste lineal.

    E = (1/m) * ||A·x − b||₂².

    Parámetros
    ----------
    A : secuencia de secuencias
        Matriz de diseño m×n.
    x : secuencia
        Coeficientes del modelo.
    b : secuencia
        Vector de observaciones de longitud m.

    Retorna
    -------
    float
        Error cuadrático medio.
    """
    A_mat = np.asarray(A, dtype=float)
    x_vec = np.asarray(x, dtype=float)
    b_vec = np.asarray(b, dtype=float)
    residual = A_mat @ x_vec - b_vec
    mse = np.mean(residual**2)
    return float(mse)


if __name__ == "__main__":
    # Ejemplo rápido: ajuste de una línea a puntos con ruido
    import random
    xs = [i for i in range(5)]
    ys = [2 * x + 1 + (random.random() - 0.5) for x in xs]
    A = [[x, 1] for x in xs]
    print("Datos:", list(zip(xs, ys)))
    sol, cond = normal_equations(A, ys)
    print("Coeficientes normal equations:", sol, "condición:", cond)
    sol_svd, cond_svd = svd_least_squares(A, ys)
    print("Coeficientes SVD:", sol_svd, "condición:", cond_svd)