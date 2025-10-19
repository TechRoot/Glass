"""
Estimación de homografías planas mediante el método DLT normalizado.

Se proporcionan funciones para calcular una homografía 3×3 a partir de un
conjunto de correspondencias de puntos y para aplicar la homografía a
nuevos puntos.
"""
from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple


def _normalize_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normaliza un conjunto de puntos 2D para mejorar la estabilidad numérica.

    El procedimiento centra los puntos en el origen y escala de manera que la
    distancia media al origen sea sqrt(2).

    Args:
        pts: array de forma (N, 2).

    Returns:
        Una tupla (pts_norm, T) donde pts_norm es el array normalizado y T es
        la matriz de transformación 3×3 tal que pts_norm = (T @ pts_homog^T)^T.
    """
    pts = np.asarray(pts, dtype=float)
    mean = pts.mean(axis=0)
    pts_cent = pts - mean
    dists = np.linalg.norm(pts_cent, axis=1)
    mean_dist = np.mean(dists)
    scale = np.sqrt(2) / mean_dist if mean_dist > 0 else 1.0
    T = np.array([
        [scale, 0,      -scale * mean[0]],
        [0,     scale, -scale * mean[1]],
        [0,     0,      1]
    ])
    # añadir homogeneidad
    pts_hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (T @ pts_hom.T).T
    pts_norm = pts_norm[:, :2] / pts_norm[:, 2:]
    return pts_norm, T


def estimate_homography(src_pts: Iterable[Tuple[float, float]], dst_pts: Iterable[Tuple[float, float]]) -> np.ndarray:
    """Calcula la homografía 3×3 que lleva src_pts a dst_pts mediante DLT.

    Args:
        src_pts: lista de puntos de origen (x, y).
        dst_pts: lista de puntos destino (x, y).

    Returns:
        Matriz 3×3 H tal que [x_dst, y_dst, 1]^T ∝ H [x_src, y_src, 1]^T.
    """
    src = np.asarray(src_pts, dtype=float)
    dst = np.asarray(dst_pts, dtype=float)
    assert src.shape == dst.shape and src.shape[0] >= 4
    # Normalizar puntos
    src_norm, T_src = _normalize_points(src)
    dst_norm, T_dst = _normalize_points(dst)
    n = src.shape[0]
    A = []
    for i in range(n):
        x, y = src_norm[i]
        u, v = dst_norm[i]
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
    A = np.array(A, dtype=float)
    # Resolver A h = 0 usando SVD: el vector singular correspondiente al valor mínimo
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H_norm = h.reshape(3, 3)
    # Desnormalizar
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    H = H / H[2, 2]
    return H


def apply_homography(points: Iterable[Tuple[float, float]], H: np.ndarray) -> np.ndarray:
    """Aplica una homografía a una colección de puntos 2D.

    Args:
        points: iterables de (x, y).
        H: matriz 3×3 de homografía.

    Returns:
        Array de forma (N, 2) con los puntos transformados.
    """
    pts = np.asarray(list(points), dtype=float)
    n = pts.shape[0]
    pts_hom = np.hstack([pts, np.ones((n, 1))])
    trans = (H @ pts_hom.T).T
    trans = trans[:, :2] / trans[:, 2:]
    return trans
