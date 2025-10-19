"""
Funciones de apoyo para transformaciones lineales y rotaciones en 3D.

Se implementan matrices homogéneas SE(3), la conversión entre cuaterniones y
matrices de rotación y el cálculo del radio espectral para estudiar la
estabilidad de operadores discretos.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


def homogeneous_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Construye una matriz de transformación homogénea 4×4 a partir de una
    matriz de rotación 3×3 y un vector de traslación 3×1.

    Args:
        rotation: matriz 3×3 de rotación.
        translation: vector de traslación de longitud 3.

    Returns:
        Una matriz 4×4 representando una transformación en SE(3).
    """
    T = np.eye(4, dtype=float)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def apply_transform(point: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Aplica una transformación homogénea a un punto en coordenadas homogéneas.

    Args:
        point: vector (3,) o (4,) que representa un punto en coordenadas
            cartesianas; si tiene longitud 3 se considera w=1.
        transform: matriz 4×4 de transformación.

    Returns:
        El punto transformado en coordenadas cartesianas de longitud 3.
    """
    if point.shape[0] == 3:
        p = np.ones(4)
        p[:3] = point
    else:
        p = point
    result = transform @ p
    return result[:3] / result[3] if result[3] != 0 else result[:3]


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Convierte un cuaternión (w, x, y, z) a una matriz de rotación 3×3.

    El cuaternión debe ser unitario para representar una rotación pura.
    """
    w, x, y, z = q
    # normalización en caso de desviación numérica
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("El cuaternión no puede ser nulo")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),         1 - 2*(x*x + y*y)]
    ], dtype=float)


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convierte una matriz de rotación 3×3 a un cuaternión (w, x, y, z).

    Usa el algoritmo basado en el rastro de la matriz.
    """
    m = R
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=float)


def spectral_radius(matrix: np.ndarray) -> float:
    """Calcula el radio espectral (máximo valor absoluto de los autovalores)."""
    eigenvalues = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigenvalues)))
