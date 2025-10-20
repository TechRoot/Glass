"""
Funciones utilitarias para estimar y aplicar homografías planas.

Este módulo implementa un algoritmo de Transformación Lineal Directa normalizada (DLT)
para estimar una matriz de homografía 3×3 a partir de al menos cuatro correspondencias
de puntos y funciones para aplicar dicha homografía a conjuntos de puntos.

El algoritmo sigue el procedimiento estándar en visión por computador: los puntos se
normalizan para mejorar el acondicionamiento numérico, la homografía se estima
mediante descomposición en valores singulares (SVD) de la matriz normalizada, y el
resultado se desnormaliza antes de devolverlo.  Las homografías mapean puntos de un
plano a otro y se representan por matrices 3×3 definidas hasta un factor de escala.

Dependencias: NumPy ≥ 1.20

Autor: A. Alonso
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple


def _normalize_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve los puntos normalizados y la matriz de normalización correspondiente.

    Dado un conjunto de puntos 2D ``pts`` con forma (N, 2), esta función
    calcula una transformación de similitud que traslada el centroide
    de los puntos al origen y escala los puntos de manera que la distancia
    media al origen sea ``√2``.  Esta normalización ayuda a estabilizar
    la estimación de homografías.

    Parámetros
    ----------
    pts : ndarray
        Matriz de forma (N, 2) que contiene puntos 2D.

    Devuelve
    -------
    pts_norm : ndarray
        Puntos normalizados de forma (N, 3) en coordenadas homogéneas.
    T : ndarray
        Matriz de normalización 3×3 tal que ``pts_norm = T @ pts_hom``.
    """
    centroid = np.mean(pts, axis=0)
    diff = pts - centroid
    mean_dist = np.mean(np.linalg.norm(diff, axis=1))
    scale = np.sqrt(2) / mean_dist
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])
    pts_hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm = (T @ pts_hom.T).T
    return pts_norm, T


def compute_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estima una homografía plana que mapea los puntos ``src`` en ``dst``.

    Parámetros
    ----------
    src : ndarray
        Puntos de origen de forma (N, 2).  Se requieren al menos cuatro puntos no colineales.
    dst : ndarray
        Puntos de destino de forma (N, 2).  Deben corresponder a ``src``.

    Devuelve
    -------
    H : ndarray
        Matriz de homografía estimada 3×3 (no normalizada).

    Notas
    -----
    Si los puntos son casi colineales o la configuración es degenerada,
    los valores singulares pueden no estar bien separados y el resultado
    puede ser inestable.  En la práctica, aplicar una normalización
    (véase ``_normalize_points``) mejora la estabilidad【185657541199976†L26-L75】.
    """
    if src.shape != dst.shape:
        raise ValueError("Los puntos de origen y destino deben tener la misma forma")
    if src.shape[0] < 4:
        raise ValueError("Se requieren al menos cuatro correspondencias de puntos")
    # Normalizar los puntos de entrada
    src_norm, T_src = _normalize_points(src)
    dst_norm, T_dst = _normalize_points(dst)
    # Construir el sistema lineal A * h = 0
    N = src.shape[0]
    A = []
    for i in range(N):
        x, y, w = src_norm[i]
        u, v, _ = dst_norm[i]
        A.append([0, 0, 0, -w * x, -w * y, -w * w, v * x, v * y, v * w])
        A.append([w * x, w * y, w * w, 0, 0, 0, -u * x, -u * y, -u * w])
    A = np.array(A)
    # Solve via SVD
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H_norm = h.reshape(3, 3)
    # Denormalizar: H = T_dst^{-1} * H_norm * T_src
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    # normalizar para que H[2,2] = 1
    return H / H[2, 2]


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Aplica la homografía ``H`` a los puntos ``pts``.

    Parámetros
    ----------
    H : ndarray
        Matriz de homografía 3×3.
    pts : ndarray
        Puntos de forma (N, 2) que se van a transformar.

    Devuelve
    -------
    transformed : ndarray
        Puntos transformados de forma (N, 2).
    """
    pts_hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    mapped = (H @ pts_hom.T).T
    mapped /= mapped[:, 2][:, None]
    return mapped[:, :2]
