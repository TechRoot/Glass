"""
Mochila 0‑1 resuelta mediante programación dinámica.

El problema de la mochila 0‑1 consiste en elegir un subconjunto de
objetos con pesos y valores dados para maximizar el valor total sin
exceder una capacidad de peso.  Esta implementación utiliza una tabla de
programación dinámica para calcular los valores óptimos y reconstruir
los objetos seleccionados【386418027411241†L10-L35】.

Dependencias: solo la biblioteca estándar de Python
"""

from __future__ import annotations

from typing import List, Tuple


def knapsack(values: List[int], weights: List[int], capacity: int) -> Tuple[int, List[int]]:
    """Devuelve el valor máximo y los índices de los objetos seleccionados."""
    n = len(values)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    # Reconstruye la solución
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)
            w -= weights[i - 1]
    selected.reverse()
    return dp[n][capacity], selected
