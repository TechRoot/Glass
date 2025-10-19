"""Algoritmo de asignación óptima mediante el método Húngaro.

Este módulo implementa una versión simplificada del algoritmo de
asignación de costes (también conocido como método Húngaro). A partir
de una matriz de costes cuadrada o rectangular, calcula la asignación
    de filas a columnas de coste mínimo. Si la matriz no es cuadrada, se
    añaden filas o columnas ficticias con coste cero para completar un
    cuadrado.  La complejidad del algoritmo es cúbica en el número de
    filas (O(n³)).

Funciones principales:

* `hungarian(costs)` → devuelve una lista de pares (fila, columna)
  representando la asignación óptima y el coste total.
* `solve_assignment_from_csv(path)` → carga una matriz de un fichero
  CSV y aplica el algoritmo.
"""

from __future__ import annotations

from typing import List, Tuple, Sequence, Optional
import copy
import csv


def _pad_matrix(matrix: List[List[float]], value: float = 0.0) -> List[List[float]]:
    """Rellena con filas/columnas ficticias para obtener una matriz cuadrada.

    Args:
        matrix: lista de listas con los costes.
        value: valor por defecto para nuevas celdas.

    Returns:
        Una nueva matriz cuadrada con el tamaño máximo de filas y columnas.
    """
    n_rows = len(matrix)
    n_cols = max(len(r) for r in matrix) if matrix else 0
    size = max(n_rows, n_cols)
    padded = [r + [value] * (size - len(r)) for r in matrix]
    for _ in range(size - n_rows):
        padded.append([value] * size)
    return padded


def _reduce_rows_and_cols(costs: List[List[float]]) -> None:
    """Reduce filas y columnas restando los mínimos correspondientes.

    Modifica la matriz in situ restando el mínimo de cada fila y luego
    el mínimo de cada columna.
    """
    # Reducción por filas
    for i, row in enumerate(costs):
        m = min(row)
        costs[i] = [v - m for v in row]
    # Reducción por columnas
    size = len(costs)
    for j in range(size):
        col = [costs[i][j] for i in range(size)]
        m = min(col)
        for i in range(size):
            costs[i][j] -= m


def _find_zero(costs, covered_rows, covered_cols) -> Optional[Tuple[int, int]]:
    """Encuentra un cero sin cubrir. Devuelve (fila, col) o None."""
    size = len(costs)
    for i in range(size):
        if covered_rows[i]:
            continue
        for j in range(size):
            if covered_cols[j]:
                continue
            if costs[i][j] == 0:
                return (i, j)
    return None


def _make_assignment(costs: List[List[float]]) -> List[Tuple[int, int]]:
    """Implementación principal del algoritmo Húngaro.

    Args:
        costs: matriz cuadrada de costes (modificada internamente).

    Returns:
        Lista de pares (fila, columna) que representan la asignación
        óptima.  La matriz de entrada se modifica.
    """
    size = len(costs)
    # Inicialmente, ninguna fila o columna está cubierta
    covered_rows = [False] * size
    covered_cols = [False] * size
    # Matriz de marcas: 1 = estrella, 2 = prima
    marks = [[0] * size for _ in range(size)]
    # Paso 1: reducir filas y columnas
    _reduce_rows_and_cols(costs)
    # Paso 2: estrellas iniciales
    for i in range(size):
        for j in range(size):
            if costs[i][j] == 0 and not any(marks[i][k] == 1 for k in range(size)) and not any(marks[l][j] == 1 for l in range(size)):
                marks[i][j] = 1
    # Paso 3: cubrir columnas con estrellas
    for j in range(size):
        if any(marks[i][j] == 1 for i in range(size)):
            covered_cols[j] = True
    # Principal bucle
    while sum(covered_cols) < size:
        # Paso 4: buscar cero sin cubrir y primarlo
        zero = _find_zero(costs, covered_rows, covered_cols)
        while zero is None:
            # Ajustar la matriz restando el mínimo de los no cubiertos y sumando a los cubiertos
            min_uncovered = float('inf')
            for i in range(size):
                if covered_rows[i]:
                    continue
                for j in range(size):
                    if covered_cols[j]:
                        continue
                    if costs[i][j] < min_uncovered:
                        min_uncovered = costs[i][j]
            # Ajustar valores
            for i in range(size):
                for j in range(size):
                    if not covered_rows[i] and not covered_cols[j]:
                        costs[i][j] -= min_uncovered
                    elif covered_rows[i] and covered_cols[j]:
                        costs[i][j] += min_uncovered
            zero = _find_zero(costs, covered_rows, covered_cols)
        # Primo el cero encontrado
        i, j = zero
        marks[i][j] = 2
        # Si hay una estrella en la misma fila, cubrir fila y descubrir columna de la estrella
        star_col = None
        for col in range(size):
            if marks[i][col] == 1:
                star_col = col
                break
        if star_col is not None:
            covered_rows[i] = True
            covered_cols[star_col] = False
        else:
            # Construir cadena alternante
            seq = [(i, j)]
            done = False
            while not done:
                # Buscar estrella en la columna del último primo
                row_star = None
                last_col = seq[-1][1]
                for row in range(size):
                    if marks[row][last_col] == 1:
                        row_star = row
                        break
                if row_star is None:
                    done = True
                else:
                    seq.append((row_star, last_col))
                    # Buscar primo en la fila del nuevo estrella
                    col_prime = None
                    for col in range(size):
                        if marks[row_star][col] == 2:
                            col_prime = col
                            break
                    seq.append((row_star, col_prime))
            # Alternar marcas en la cadena: estrellas <-> primos
            for r, c in seq:
                if marks[r][c] == 1:
                    marks[r][c] = 0
                elif marks[r][c] == 2:
                    marks[r][c] = 1
            # Borrar todas las marcas primas y descubrir todas las filas y columnas
            for r in range(size):
                for c in range(size):
                    if marks[r][c] == 2:
                        marks[r][c] = 0
            covered_rows = [False] * size
            covered_cols = [False] * size
            # Cubrir columnas con estrellas
            for col in range(size):
                if any(marks[r][col] == 1 for r in range(size)):
                    covered_cols[col] = True
    # Extraer asignación de estrellas
    assignment = []
    for i in range(size):
        for j in range(size):
            if marks[i][j] == 1:
                assignment.append((i, j))
                break
    return assignment


def hungarian(matrix: Sequence[Sequence[float]]) -> Tuple[List[Tuple[int, int]], float]:
    """Resuelve el problema de asignación para la matriz de costes dada.

    Args:
        matrix: matriz de costes (listas o tuplas) de tamaño NxM.

    Returns:
        Par `(assignment, total_cost)` donde `assignment` es una lista de
        tuplas (fila, columna) y `total_cost` es la suma de los costes
        asignados según la entrada original.
    """
    # Copiar la matriz y rellenar a cuadrada
    cost_matrix = [list(row) for row in matrix]
    padded = _pad_matrix(cost_matrix)
    # Copiar para no modificar la original al reducir
    working = copy.deepcopy(padded)
    assignment = _make_assignment(working)
    # Filtrar asignación a dimensión original
    assignment = [(r, c) for r, c in assignment if r < len(matrix) and c < len(matrix[0])]
    total = sum(matrix[r][c] for r, c in assignment)
    return assignment, total


def solve_assignment_from_csv(path: str) -> Tuple[List[Tuple[int, int]], float]:
    """Lee una matriz de costes desde un CSV y aplica el algoritmo Húngaro.

    Args:
        path: ruta al archivo CSV. Cada fila representa una fila de la matriz y
              las columnas deben ser separadas por comas.

    Returns:
        Par `(assignment, total_cost)` con la solución de asignación.
    """
    matrix: List[List[float]] = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            matrix.append([float(v) for v in row])
    return hungarian(matrix)