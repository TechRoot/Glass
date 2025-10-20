"""
Algoritmo húngaro para problemas de asignación.

Este módulo implementa el algoritmo húngaro (también conocido como algoritmo de
Kuhn–Munkres) para resolver el problema de asignación en tiempo O(n³).  Dada
una matriz de costes cuadrada, encuentra el emparejamiento de coste mínimo
entre filas y columnas.  Se incluye una función auxiliar para cargar matrices de
costes desde archivos CSV【139678897884343†L25-L208】.

Dependencias: NumPy ≥1.20, csv
"""

from __future__ import annotations

import csv
from typing import List, Tuple

import numpy as np


def load_cost_matrix(path: str) -> np.ndarray:
    """Carga una matriz de costes desde un archivo CSV."""
    with open(path, newline='') as f:
        reader = csv.reader(f)
        rows = [[float(cell) for cell in row] for row in reader]
    return np.array(rows)


def hungarian_algorithm(cost: np.ndarray) -> Tuple[List[int], float]:
    """Resuelve el problema de asignación para una matriz de costes.

    Devuelve una tupla ``(asignación, coste_total)`` donde ``asignación``
    asocia cada fila con la columna elegida y ``coste_total`` es la suma
    de los costes seleccionados【139678897884343†L25-L208】.
    """
    orig_cost = cost.copy()
    cost = cost.copy()
    n = cost.shape[0]
    # Paso 1: Restar mínimos de fila
    for i in range(n):
        cost[i] -= cost[i].min()
    # Paso 2: Restar mínimos de columna
    for j in range(n):
        cost[:, j] -= cost[:, j].min()
    # Empiezo de la cobertura de ceros
    starred = np.zeros_like(cost, dtype=bool)
    row_cover = np.zeros(n, dtype=bool)
    col_cover = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if cost[i, j] == 0 and not row_cover[i] and not col_cover[j]:
                starred[i, j] = True
                row_cover[i] = True
                col_cover[j] = True
    row_cover[:] = False
    col_cover[:] = False
    # Helpers
    def cover_columns_with_stars():
        for j in range(n):
            if np.any(starred[:, j]):
                col_cover[j] = True

    def find_a_zero():
        for i in range(n):
            if not row_cover[i]:
                for j in range(n):
                    if cost[i, j] == 0 and not col_cover[j]:
                        return i, j
        return None

    def find_star_in_row(i):
        for j in range(n):
            if starred[i, j]:
                return j
        return None

    def find_star_in_col(j):
        for i in range(n):
            if starred[i, j]:
                return i
        return None

    def find_prime_in_row(i):
        for j in range(n):
            if primed[i, j]:
                return j
        return None

    # Bucle principal
    cover_columns_with_stars()
    while True:
        if np.sum(col_cover) == n:
            break  # Todas las columnas están cubiertas
        # Paso 3: Marcar ceros no cubiertos
        primed = np.zeros_like(cost, dtype=bool)
        while True:
            zero = find_a_zero()
            if zero is None:
                # no se encontraron ceros no cubiertos
                # Ajustar la matriz de costes   
                min_uncovered = np.min(cost[~row_cover[:, None] & ~col_cover])
                cost[~row_cover[:, None]] -= min_uncovered
                cost[:, col_cover] += min_uncovered
                zero = find_a_zero()
            i, j = zero
            primed[i, j] = True
            star_col = find_star_in_row(i)
            if star_col is None:
                # Paso 4: Aumentar a lo largo del camino alternante
                seq = [(i, j)]
                while True:
                    star_row = find_star_in_col(seq[-1][1])
                    if star_row is None:
                        break
                    seq.append((star_row, seq[-1][1]))
                    prime_col = find_prime_in_row(seq[-1][0])
                    seq.append((seq[-1][0], prime_col))
                for r, c in seq:
                    starred[r, c] = not starred[r, c]
                primed[:, :] = False
                row_cover[:] = False
                col_cover[:] = False
                cover_columns_with_stars()
                break
            else:
                row_cover[i] = True
                col_cover[star_col] = False
        #fin 
    assignment = [-1] * n
    for i in range(n):
        j = np.where(starred[i])[0][0]
        assignment[i] = int(j)
    total_cost = float(sum(orig_cost[i, assignment[i]] for i in range(n)))
    return assignment, total_cost
