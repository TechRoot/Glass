"""Módulo de interpolación de polinomios.

Este archivo contiene funciones para construir y evaluar polinomios de
interpolación usando distintos enfoques: la forma de Lagrange,
la forma de Newton con diferencias divididas y la evaluación
baricéntrica. También permite resolver el sistema de Vandermonde
para obtener directamente los coeficientes del polinomio que pasa
exactamente por un conjunto dado de puntos.

Las implementaciones utilizan NumPy para operaciones matriciales
cuando resulta más conveniente.  Se prioriza la claridad y la
trazabilidad por encima del rendimiento bruto, dado que los
ejemplos están orientados a demostraciones y pruebas.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def lagrange_basis(x_values: Sequence[float], i: int, x: float) -> float:
    """Calcula el valor de la i‑ésima base de Lagrange en el punto x.

    La base L_i(x) se define como el producto sobre j≠i de
    (x - x_j) / (x_i - x_j).  Este valor se usa tanto para evaluar
    directamente el polinomio de Lagrange como para obtener los
    coeficientes de manera explícita.

    Parámetros
    ----------
    x_values : secuencia de float
        Valores de abscisas de los puntos de interpolación.
    i : int
        Índice del término de base.
    x : float
        Punto en el que se evalúa la base.

    Retorna
    -------
    float
        Valor de la base L_i(x).
    """
    xi = x_values[i]
    prod = 1.0
    for j, xj in enumerate(x_values):
        if j == i:
            continue
        denom = xi - xj
        if denom == 0:
            raise ValueError("Los valores x deben ser distintos")
        prod *= (x - xj) / denom
    return prod


def lagrange_interpolate(x_values: Sequence[float], y_values: Sequence[float], x: float) -> float:
    """Evalúa el polinomio de interpolación de Lagrange en un punto dado.

    Esta función suma los valores y_i multiplicados por sus bases L_i(x).
    Es adecuada para un número moderado de puntos, ya que la complejidad
    es O(n^2) por evaluación. Para evaluar muchos puntos, considere
    `barycentric_eval`.

    Parámetros
    ----------
    x_values : secuencia de float
        Abscisas de los puntos de interpolación.
    y_values : secuencia de float
        Ordenadas correspondientes.
    x : float
        Punto en el que se desea evaluar el polinomio.

    Retorna
    -------
    float
        Valor del polinomio interpolado en x.
    """
    if len(x_values) != len(y_values):
        raise ValueError("x_values y y_values deben tener la misma longitud")
    result = 0.0
    for i in range(len(x_values)):
        result += y_values[i] * lagrange_basis(x_values, i, x)
    return result


def barycentric_weights(x_values: Sequence[float]) -> List[float]:
    """Calcula los pesos baricéntricos para interpolación racional de Lagrange.

    Los pesos w_i se definen como 1 / ∏_{j≠i} (x_i - x_j).  Estos pesos
    permiten evaluar el polinomio de interpolación en O(n) por punto,
    reutilizando los mismos pesos para múltiples evaluaciones.

    Parámetros
    ----------
    x_values : secuencia de float
        Abscisas de los puntos de interpolación.

    Retorna
    -------
    List[float]
        Lista de pesos baricéntricos.
    """
    n = len(x_values)
    w = [1.0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = x_values[i] - x_values[j]
            if diff == 0:
                raise ValueError("Los valores x deben ser distintos")
            w[i] /= diff
    return w


def barycentric_eval(
    x_values: Sequence[float],
    y_values: Sequence[float],
    weights: Sequence[float] | None,
    x: float,
) -> float:
    """Evalúa el polinomio usando la forma baricéntrica.

    Si `x` coincide con uno de los valores de `x_values`, se devuelve el
    valor correspondiente de `y_values` directamente para evitar división por
    cero. En caso contrario, se calcula la suma de w_i y_i / (x - x_i)
    normalizada por la suma de w_i / (x - x_i).

    Parámetros
    ----------
    x_values : secuencia de float
        Abscisas de los puntos de interpolación.
    y_values : secuencia de float
        Ordenadas correspondientes.
    weights : secuencia de float o None
        Pesos baricéntricos precomputados. Si es None, se calcularán.
    x : float
        Punto en el que se desea evaluar el polinomio.

    Retorna
    -------
    float
        Valor del polinomio interpolado en x.
    """
    if len(x_values) != len(y_values):
        raise ValueError("x_values y y_values deben tener la misma longitud")
    if weights is None:
        weights = barycentric_weights(x_values)
    # Comprobar si x coincide con algún nodo
    for xi, yi in zip(x_values, y_values):
        if x == xi:
            return yi
    num = 0.0
    den = 0.0
    for xi, yi, wi in zip(x_values, y_values, weights):
        diff = x - xi
        term = wi / diff
        num += term * yi
        den += term
    return num / den


def newton_divided_differences(x_values: Sequence[float], y_values: Sequence[float]) -> List[float]:
    """Calcula los coeficientes de Newton mediante diferencias divididas.

    Devuelve una lista de coeficientes c_i tal que el polinomio se
    puede evaluar como:

      P(x) = c_0 + c_1(x - x_0) + c_2(x - x_0)(x - x_1) + ...

    Parámetros
    ----------
    x_values : secuencia de float
        Abscisas de los puntos de interpolación.
    y_values : secuencia de float
        Ordenadas correspondientes.

    Retorna
    -------
    List[float]
        Lista de coeficientes de Newton.
    """
    n = len(x_values)
    if n != len(y_values):
        raise ValueError("x_values y y_values deben tener la misma longitud")
    coef = list(y_values)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            denom = x_values[i] - x_values[i - j]
            if denom == 0:
                raise ValueError("Los valores x deben ser distintos")
            coef[i] = (coef[i] - coef[i - 1]) / denom
    return coef


def newton_eval(x_values: Sequence[float], coef: Sequence[float], x: float) -> float:
    """Evalúa un polinomio en forma de Newton en un punto dado.

    Parámetros
    ----------
    x_values : secuencia de float
        Puntos de interpolación originales.
    coef : secuencia de float
        Coeficientes devueltos por `newton_divided_differences`.
    x : float
        Punto en el que se evalúa el polinomio.

    Retorna
    -------
    float
        Valor del polinomio en x.
    """
    n = len(coef)
    result = coef[-1]
    for i in range(n - 2, -1, -1):
        result = result * (x - x_values[i]) + coef[i]
    return result


def vandermonde_coeffs(x_values: Sequence[float], y_values: Sequence[float]) -> np.ndarray:
    """Resuelve el sistema de Vandermonde para obtener los coeficientes del polinomio.

    Construye una matriz de Vandermonde V donde V[i,j] = x_i^j y resuelve
    V * c = y para c. Se usa la rutina de NumPy que internamente aplica
    algoritmos robustos como la factorización QR o la SVD según el caso.

    Parámetros
    ----------
    x_values : secuencia de float
        Abscisas de los puntos de interpolación.
    y_values : secuencia de float
        Ordenadas correspondientes.

    Retorna
    -------
    np.ndarray
        Vector de coeficientes c, donde c[j] es el coeficiente de x^j.
    """
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    if x_arr.ndim != 1:
        raise ValueError("x_values debe ser un vector")
    n = x_arr.size
    V = np.vander(x_arr, N=n, increasing=True)
    # Resolver el sistema V c = y
    c, residuals, rank, s = np.linalg.lstsq(V, y_arr, rcond=None)
    return c


if __name__ == "__main__":
    # Pequeño ejemplo de uso interactivo
    xs = [0, 1, 2]
    ys = [1, 3, 2]
    print("Polinomio de Lagrange en x=1.5:", lagrange_interpolate(xs, ys, 1.5))
    w = barycentric_weights(xs)
    print("Evaluación baricéntrica en x=1.5:", barycentric_eval(xs, ys, w, 1.5))
    coeffs = newton_divided_differences(xs, ys)
    print("Evaluación de Newton en x=1.5:", newton_eval(xs, coeffs, 1.5))
    vand_coeffs = vandermonde_coeffs(xs, ys)
    print("Coeficientes de Vandermonde:", vand_coeffs)