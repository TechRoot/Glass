"""Módulo para pruebas de corrección y seguridad.

Este módulo agrupa funciones destinadas a verificar la terminación de
algoritmos y máquinas de estados, la conservación de medidas en
transformaciones geométricas, la optimalidad de soluciones obtenidas por
algoritmos de asignación, la validez de inversos modulares y la
integridad de códigos CRC.  Su objetivo es servir como base para
pruebas de regresión y análisis de seguridad en sistemas industriales.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from . import assignment
from . import crc


def check_nilpotent_and_steps(matrix: Sequence[Sequence[int]]) -> Tuple[bool, int | None]:
    """Comprueba si una matriz cuadrada es nilpotente y devuelve el exponente mínimo.

    Una matriz A es nilpotente si existe algún k tal que A^k = 0.  Para
    máquinas de estados finitos, una matriz de adyacencia nilpotente
    indica que no existen ciclos y que el autómata termina en un número
    finito de pasos.  Se calcula A^p sucesivamente hasta alcanzar la
    matriz nula o hasta un límite de n pasos, donde n es el tamaño de la
    matriz.

    Parámetros
    ----------
    matrix : secuencia de secuencias de enteros (0/1)
        Matriz de adyacencia de tamaño n×n.

    Retorna
    -------
    Tuple[bool, int | None]
        Par (es_nilpotente, pasos). Si la matriz es nilpotente, `pasos`
        indica el menor exponente k tal que A^k = 0; en caso contrario,
        `pasos` es None.
    """
    A = np.asarray(matrix, dtype=int)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("La matriz debe ser cuadrada")
    # Inicializar potencia
    power = A.copy()
    for k in range(1, n + 1):
        if not power.any():
            return True, k
        power = power @ A
    return False, None


def check_rotation_invariants(matrix: Sequence[Sequence[float]], tol: float = 1e-6) -> bool:
    """Verifica que una matriz de rotación conserva normas y orientación.

    Comprueba que R^T R = I (ortogonalidad) y que det(R) ≈ 1.  Estas
    condiciones garantizan que la transformación es rígida, conservando
    distancias y orientaciones (sin reflexiones).

    Parámetros
    ----------
    matrix : secuencia de secuencias de float
        Matriz 3×3 candidata a ser una rotación.
    tol : float, opcional
        Tolerancia para la comparación numérica.

    Retorna
    -------
    bool
        True si la matriz es ortogonal y tiene determinante cercano a 1.
    """
    R = np.asarray(matrix, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("La matriz debe ser 3x3")
    RtR = R.T @ R
    I = np.eye(3)
    return np.allclose(RtR, I, atol=tol) and abs(np.linalg.det(R) - 1.0) < tol


def verify_primal_dual(cost: Sequence[Sequence[float]], assignment_pairs: Sequence[Tuple[int, int]], tol: float = 1e-6) -> bool:
    """Verifica la optimalidad primal–dual de una asignación.

    Esta función comprueba que el coste de la asignación coincide con el
    coste mínimo calculado mediante el método Húngaro y que no existe
    asignación alternativa con coste inferior.  Si ambas condiciones se
    cumplen, la solución satisface las condiciones primal–dual y se
    considera óptima.

    Parámetros
    ----------
    cost : secuencia de secuencias
        Matriz de costes m×n (puede ser rectangular).
    assignment_pairs : secuencia de pares (fila, columna)
        Asignación propuesta. Cada par indica la selección de un
        elemento en la fila y columna correspondientes.
    tol : float, opcional
        Tolerancia para comparar costes.

    Retorna
    -------
    bool
        True si la asignación es óptima y cumple la condición
        primal–dual; False en caso contrario.
    """
    # Calcular coste propuesto
    total_cost = 0.0
    for i, j in assignment_pairs:
        total_cost += cost[i][j]
    # Calcular asignación óptima mediante el algoritmo de asignación
    opt_assign, opt_cost = assignment.hungarian(cost)
    # Comparar costes
    return abs(total_cost - opt_cost) <= tol


def verify_inverse_mod(a: int, m: int, inv: int) -> bool:
    """Comprueba que `inv` es el inverso modular de `a` módulo `m`.

    Se verifica simplemente que (a * inv) mod m = 1.
    """
    if m <= 1:
        raise ValueError("El módulo debe ser mayor que 1")
    return (a * inv) % m == 1


def verify_crc_error(data: bytes | Iterable[int], crc_value: int, poly: int = 0x07) -> bool:
    """Comprueba si un CRC coincide con los datos o detecta un error.

    Devuelve ``True`` si el CRC es correcto (es decir, la concatenación de
    datos y CRC pasa la verificación) y ``False`` en caso contrario.  Para la
    verificación se reutilizan las funciones disponibles en el módulo
    ``crc.py``.
    """
    return crc.verify_crc8(data, crc_value, poly=poly)