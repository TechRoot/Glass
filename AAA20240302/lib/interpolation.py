"""
Rutinas de interpolación polinómica.

Este módulo proporciona esquemas clásicos de interpolación para aproximar
una función conocida en puntos discretos: interpolación de Lagrange,
diferencias divididas de Newton con evaluación incremental y la forma
baricéntrica para una mayor estabilidad numérica.  También incluye una
función para resolver sistemas de Vandermonde de forma directa, aunque suelen
estar mal condicionados【265548766805478†L22-L249】.

Dependencias: NumPy ≥ 1.20
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


def lagrange_interpolate(x_points: List[float], y_points: List[float], x: float) -> float:
    """Evalúa el polinomio de interpolación de Lagrange en x."""
    n = len(x_points)
    total = 0.0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        total += term
    return total


def barycentric_weights(x_points: List[float]) -> List[float]:
    """Calcula los pesos baricéntricos para una evaluación polinómica estable."""
    n = len(x_points)
    w = [1.0] * n
    for j in range(n):
        for i in range(n):
            if i != j:
                w[j] /= (x_points[j] - x_points[i])
    return w


def barycentric_interpolate(x_points: List[float], y_points: List[float], w: List[float], x: float) -> float:
    """Evalúa el polinomio de interpolación utilizando pesos baricéntricos."""
    numer = 0.0
    denom = 0.0
    for xi, yi, wi in zip(x_points, y_points, w):
        if np.isclose(x, xi):
            return yi
        temp = wi / (x - xi)
        numer += temp * yi
        denom += temp
    return numer / denom


def newton_divided_differences(x_points: List[float], y_points: List[float]) -> List[float]:
    """Calcula los coeficientes de las diferencias divididas de Newton."""
    n = len(x_points)
    coef = y_points.copy()
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x_points[i] - x_points[i - j])
    return coef


def newton_evaluate(coef: List[float], x_points: List[float], x: float) -> float:
    """Evalúa un polinomio en forma de Newton en x."""
    n = len(coef)
    result = coef[-1]
    for i in range(n - 2, -1, -1):
        result = result * (x - x_points[i]) + coef[i]
    return result


def solve_vandermonde(x_points: List[float], y_points: List[float]) -> List[float]:
    """Resuelve el sistema de Vandermonde para obtener los coeficientes del polinomio."""
    V = np.vander(x_points, increasing=True)
    coef = np.linalg.solve(V, y_points)
    return coef.tolist()
