"""
Implementación del algoritmo de Kabsch y su variante incremental.

El algoritmo de Kabsch calcula la rotación óptima (en el sentido de mínimos cuadrados)
que alinea dos conjuntos de puntos 3D correspondientes.  Este módulo incluye una
función para la solución clásica de Kabsch y otra para actualizar una alineación
existente de forma incremental a medida que se añaden más datos【55796530636635†L14-L83】.

Dependencias: NumPy ≥ 1.20
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Devuelve la matriz de rotación ``R`` y el vector de traslación ``t`` que alinean P con Q.

    Parámetros
    ----------
    P : ndarray
        Matriz de forma (N, 3) que representa los puntos de origen.
    Q : ndarray
        Matriz de forma (N, 3) que representa los puntos de destino.

    Devuelve
    -------
    R : ndarray
        Matriz de rotación 3×3.
    t : ndarray
        Vector de traslación de 3 elementos.
    """
    if P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError("P y Q deben tener la misma forma (N, 3)")
    # Calcula los centroides
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    # Centrar los puntos
    P_centered = P - cP
    Q_centered = Q - cQ
    # Computa la matriz de covarianza
    H = P_centered.T @ Q_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # asegurar (determinante +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cQ - R @ cP
    return R, t


def incremental_kabsch(prev_P: np.ndarray, prev_Q: np.ndarray,
                       new_P: np.ndarray, new_Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Actualiza la alineación de Kabsch de manera incremental.

    Dados conjuntos de puntos previamente alineados ``prev_P`` y ``prev_Q`` y un
    nuevo lote de correspondencias ``new_P`` y ``new_Q``, esta función concatena los
    conjuntos de datos y vuelve a calcular la solución de Kabsch.  Este enfoque
    incremental es eficiente cuando el número de puntos crece con el tiempo【55796530636635†L14-L83】.
    """
    P_concat = np.vstack([prev_P, new_P])
    Q_concat = np.vstack([prev_Q, new_Q])
    return kabsch(P_concat, Q_concat)
