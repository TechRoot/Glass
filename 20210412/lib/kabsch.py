"""
Implementación del algoritmo de Kabsch para alineamiento rígido de nubes de
puntos en 3D, junto con una actualización incremental de centroides y
covarianza.

El algoritmo calcula la rotación y traslación óptimas que minimizan el error
cuadrático medio entre dos conjuntos de puntos correspondientes.
"""
from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple, Optional


def kabsch(P: Iterable[Iterable[float]], Q: Iterable[Iterable[float]]) -> Tuple[np.ndarray, np.ndarray, float]:
    """Calcula la rotación y traslación óptimas entre dos nubes de puntos.

    Args:
        P: lista o array de puntos origen de forma (N, 3).
        Q: lista o array de puntos destino de forma (N, 3).

    Returns:
        Una tupla (R, t, rmsd) donde R es la matriz de rotación 3×3, t el
        vector de traslación 3×1 y rmsd el error cuadrático medio después del
        alineamiento.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    assert P.shape == Q.shape and P.shape[0] >= 3
    # calcular centroides
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    # centrar
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    # covarianza
    H = P_centered.T @ Q_centered
    # descomposición SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # corrección de reflexión
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_Q - R @ centroid_P
    # calcular RMSD
    aligned = (R @ P_centered.T).T
    diff = aligned - Q_centered
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return R, t, float(rmsd)


class KabschIncremental:
    """Clase para actualizar centroides y covarianza de forma incremental."""
    def __init__(self):
        self.n = 0
        self.centroid_P = np.zeros(3)
        self.centroid_Q = np.zeros(3)
        self.H = np.zeros((3, 3))

    def add_points(self, P: Iterable[Iterable[float]], Q: Iterable[Iterable[float]]):
        P = np.asarray(P, dtype=float)
        Q = np.asarray(Q, dtype=float)
        for p, q in zip(P, Q):
            self.n += 1
            # actualizar centroides
            delta_P = p - self.centroid_P
            delta_Q = q - self.centroid_Q
            self.centroid_P += delta_P / self.n
            self.centroid_Q += delta_Q / self.n
            # actualizar covarianza
            self.H += np.outer(p - self.centroid_P, q - self.centroid_Q)

    def get_transform(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.n < 3:
            return None
        # SVD en covarianza
        U, S, Vt = np.linalg.svd(self.H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = self.centroid_Q - R @ self.centroid_P
        return R, t
