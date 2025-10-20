"""
Funciones para trabajar con transformaciones 3D, cuaterniones y estabilidad.

Este módulo reúne varias funciones auxiliares utilizadas en los scripts de calibración.
Incluye la conversión entre matrices de rotación y cuaterniones, la construcción de
matrices de transformación homogénea, la composición de transformaciones,
la actualización incremental de centroides y covarianzas y la comprobación del radio
espectral de una matriz para evaluar la estabilidad.

Dependencias: NumPy ≥ 1.20

Autor: Alejandro Alonso
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convierte una matriz de rotación 3×3 en un cuaternión unitario.

    El cuaternión se devuelve en el orden ``(w, x, y, z)``.
    """
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convierte un cuaternión unitario ``(w, x, y, z)`` en una matriz de rotación 3×3."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)],
    ])


def homogeneous_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Construye una matriz de transformación homogénea a partir de una rotación y una traslación.

    Parámetros
    ----------
    R : ndarray
        Matriz de rotación 3×3.
    t : ndarray
        Vector de traslación de 3 elementos.

    Devuelve
    -------
    T : ndarray
        Matriz de transformación homogénea 4×4.
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def compose_transforms(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """Compone dos transformaciones homogéneas mediante multiplicación de matrices."""
    return T1 @ T2


def update_centroid_covariance(prev_centroid: np.ndarray, prev_cov: np.ndarray,
                               new_point: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Actualiza de manera incremental el centroide y la matriz de covarianza.

    Utiliza fórmulas en línea: con ``n`` puntos previos, ``prev_centroid`` y
    ``prev_cov`` representan el centroide y la covarianza insesgada.
    Dado un nuevo punto de datos, se devuelven el nuevo centroide y la
    covarianza actualizada.
    """
    if n < 1:
        raise ValueError("n must be at least 1 for incremental update")
    new_centroid = prev_centroid + (new_point - prev_centroid) / (n + 1)
    # Update covariance using Welford's method
    delta_prev = new_point - prev_centroid
    delta_new = new_point - new_centroid
    new_cov = prev_cov + np.outer(delta_prev, delta_prev) - np.outer(delta_new, delta_new)
    return new_centroid, new_cov


def spectral_radius(matrix: np.ndarray) -> float:
    """Calcula el radio espectral (máximo valor propio en valor absoluto) de una matriz.

    El radio espectral se utiliza como criterio de estabilidad: para un sistema
    lineal ``x_{k+1} = A x_k``, el sistema es estable si el radio espectral de
    ``A`` es estrictamente menor que 1【889486680040945†L13-L106】.
    """
    eigenvalues = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigenvalues)))
