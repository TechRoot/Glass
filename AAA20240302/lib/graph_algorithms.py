"""
Algoritmos y utilidades de teoría de grafos.

Este módulo implementa varios algoritmos fundamentales de grafos: construcción
de matrices de adyacencia y laplacianas, cálculo del valor de Fiedler
(conectividad algebraica) mediante valores propios, algoritmo de Dijkstra para
camino mínimo desde una sola fuente, ordenación topológica para grafos
acíclicos dirigidos y centralidad de vector propio.

Dependencias: NumPy ≥ 1.20

Autor: A. Alonso
"""

from __future__ import annotations

import heapq
from typing import Dict, Iterable, List, Tuple

import numpy as np


def adjacency_matrix(edges: List[Tuple[int, int]], n: int) -> np.ndarray:
    """Construye una matriz de adyacencia n×n para un grafo no dirigido."""
    A = np.zeros((n, n), dtype=int)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
    return A


def laplacian_matrix(A: np.ndarray) -> np.ndarray:
    """Calcula la matriz laplaciana L = D − A."""
    deg = np.diag(np.sum(A, axis=1))
    return deg - A


def fiedler_value(L: np.ndarray) -> float:
    """Calcula el valor de Fiedler (segundo valor propio más pequeño)【722661097491708†L19-L155】."""
    eigs = np.linalg.eigvalsh(L)
    eigs = np.sort(eigs)
    return float(eigs[1]) if len(eigs) > 1 else 0.0


def dijkstra(adj: Dict[int, List[Tuple[int, float]]], source: int) -> Dict[int, float]:
    """Devuelve las distancias de los caminos más cortos desde ``source`` usando el algoritmo de Dijkstra."""
    dist = {v: float('inf') for v in adj}
    dist[source] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            alt = d + w
            if alt < dist[v]:
                dist[v] = alt
                heapq.heappush(pq, (alt, v))
    return dist


def topological_sort(adj: Dict[int, List[int]]) -> List[int]:
    """Realiza una ordenación topológica de un DAG dada una lista de adyacencia."""
    indeg = {u: 0 for u in adj}
    for u in adj:
        for v in adj[u]:
            indeg[v] = indeg.get(v, 0) + 1
    queue = [u for u, d in indeg.items() if d == 0]
    order = []
    while queue:
        u = queue.pop(0)
        order.append(u)
        for v in adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    if len(order) != len(indeg):
        raise ValueError("Ciclo encontrado: el grafo no es un DAG.")
    return order


def eigenvector_centrality(adj: Dict[int, List[int]], tol: float = 1e-6, max_iter: int = 100) -> Dict[int, float]:
    """Calcula la centralidad de vector propio de un grafo utilizando el método de la potencia."""
    nodes = list(adj.keys())
    n = len(nodes)
    x = np.ones(n)
    A = np.zeros((n, n))
    node_index = {node: i for i, node in enumerate(nodes)}
    for u, vs in adj.items():
        for v in vs:
            A[node_index[u], node_index[v]] = 1
    for _ in range(max_iter):
        x_new = A @ x
        x_new /= np.linalg.norm(x_new)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    centrality = {nodes[i]: float(x[i]) for i in range(n)}
    return centrality
