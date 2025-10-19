"""Knapsack 0-1 (programación dinámica).

Uso previsto: selección óptima de remanentes/piezas para pedidos con
capacidad limitada (peso/volumen/tiempo), como parte de un flujo de
planificación. Este módulo es autónomo y no asume dominio concreto.

Complejidad: O(n*W) en tiempo y O(n*W) en memoria (tabla completa).
"""
from typing import List, Tuple

def knapsack_01(weights: List[int], values: List[float], capacity: int) -> Tuple[float, List[int]]:
    if len(weights) != len(values):
        raise ValueError("weights y values deben tener la misma longitud")
    n = len(weights)
    W = capacity
    # DP[v][w] = valor óptimo usando items <= v con capacidad w
    DP = [[0.0]*(W+1) for _ in range(n+1)]
    for i in range(1, n+1):
        w_i = weights[i-1]
        val = values[i-1]
        for w in range(W+1):
            best = DP[i-1][w]
            if w_i <= w:
                cand = DP[i-1][w-w_i] + val
                if cand > best:
                    best = cand
            DP[i][w] = best
    # Reconstrucción de solución
    sel: List[int] = []
    w = W
    for i in range(n, 0, -1):
        if DP[i][w] != DP[i-1][w]:
            sel.append(i-1)
            w -= weights[i-1]
    sel.reverse()
    return DP[n][W], sel

if __name__ == "__main__":
    # Ejemplo mínimo
    weights = [3, 4, 2, 5]
    values  = [9.0, 6.0, 5.0, 7.5]
    cap = 7
    v, sel = knapsack_01(weights, values, cap)
    print("valor_optimo=", v, "seleccion=", sel)
