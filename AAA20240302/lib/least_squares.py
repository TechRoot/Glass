"""
Regresión lineal y solucionadores de mínimos cuadrados.

Este módulo proporciona funciones para resolver sistemas lineales sobredeterminados
``A x = b`` mediante diferentes técnicas: las ecuaciones normales (con
regularización opcional de Tikhonov), la descomposición en valores singulares (SVD)
y la descomposición QR.  También incluye una función para calcular el error
cuadrático medio (MSE) de una solución【869941161063105†L19-L116】.

Dependencias: NumPy ≥ 1.20
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def normal_eq_solve(A: np.ndarray, b: np.ndarray, lam: float = 0.0) -> np.ndarray:
    """Resuelve el sistema lineal utilizando las ecuaciones normales.

    Parámetros
    ----------
    A : ndarray
        Matriz de forma (m, n).
    b : ndarray
        Vector de forma (m,) o (m, 1).
    lam : float, opcional
        Parámetro de regularización λ ≥ 0.  Cuando λ > 0 el método resuelve
        ``(A^T A + λ I) x = A^T b`` (regresión de cresta), lo que puede
        mejorar el acondicionamiento【869941161063105†L19-L116】.

    Devuelve
    -------
    x : ndarray
        Vector solución de forma (n,).
    """
    A = np.asarray(A)
    b = np.asarray(b).reshape(-1, 1)
    m, n = A.shape
    ATA = A.T @ A
    if lam > 0:
        ATA += lam * np.eye(n)
    ATb = A.T @ b
    x = np.linalg.solve(ATA, ATb)
    return x.flatten()


def svd_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resuelve el sistema lineal utilizando la descomposición en valores singulares (SVD)."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    c = U.T @ b
    w = c / s
    x = Vt.T @ w
    return x


def qr_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resuelve el sistema lineal utilizando la descomposición QR."""
    Q, R = np.linalg.qr(A)
    return np.linalg.solve(R, Q.T @ b)


def mean_squared_error(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """Calcula el error cuadrático medio del modelo ``A x`` respecto a ``b``."""
    residual = A @ x - b
    return float(np.mean(residual ** 2))
