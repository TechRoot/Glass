"""
Algoritmos de teoría de grafos para análisis y recorrido.

Este módulo implementa funciones para construir matrices de adyacencia y
Laplacianas a partir de grafos representados como diccionarios, calcular
propiedades espectrales como el valor de Fiedler, encontrar caminos mínimos
mediante el algoritmo de Dijkstra, ordenar topológicamente grafos dirigidos
acíclicos y calcular la centralidad por eigenvector mediante el método de
la potencia.  Las funciones se diseñan para grafos pequeños y medianos con
pesos no negativos.
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Tuple, Any, Optional

import numpy as np


def adjacency_matrix(graph: Dict[Any, List[Tuple[Any, float]]]) -> Tuple[np.ndarray, Dict[Any, int]]:
    """Genera la matriz de adyacencia a partir de un grafo en formato lista de adyacencia.

    Args:
        graph: diccionario de nodo → lista de (vecino, peso).

    Returns:
        Una tupla (A, index) donde A es la matriz de adyacencia (numpy.ndarray) y
        index es un mapa de nodo a índice de fila/columna.
    """
    nodes = list(graph.keys())
    idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=float)
    for u, edges in graph.items():
        for v, w in edges:
            A[idx[u], idx[v]] = w
    return A, idx


def laplacian_matrix(graph: Dict[Any, List[Tuple[Any, float]]]) -> np.ndarray:
    """Calcula la matriz Laplaciana (L = D - A) de un grafo sin orientación.

    Para grafos no ponderados y no dirigidos, se asume peso 1 por arista. Para
    grafos dirigidos o ponderados se consideran los pesos como simétricos al
    calcular los grados. Si el grafo es dirigido, la Laplaciana resultante
    corresponde al grafo subyacente no dirigido.
    """
    A, idx = adjacency_matrix(graph)
    # simetrizar para cálculo de grados
    A_sym = A + A.T
    degrees = np.sum(A_sym > 0, axis=1) if not np.any(A_sym - A_sym.astype(bool)) else np.sum(A_sym, axis=1)
    D = np.diag(degrees)
    L = D - (A_sym if np.any(A_sym - A_sym.astype(bool)) else (A_sym > 0).astype(float))
    return L


def fiedler_value(graph: Dict[Any, List[Tuple[Any, float]]]) -> float:
    """Devuelve el valor de Fiedler (segundo autovalor más pequeño) de la Laplaciana.
    """
    L = laplacian_matrix(graph)
    eigvals = np.linalg.eigvalsh(L)
    eigvals = np.sort(eigvals)
    return float(eigvals[1]) if len(eigvals) > 1 else 0.0


def dijkstra(graph: Dict[Any, List[Tuple[Any, float]]], source: Any) -> Tuple[Dict[Any, float], Dict[Any, Optional[Any]]]:
    """Algoritmo de Dijkstra para encontrar distancias mínimas desde `source`.

    Args:
        graph: diccionario nodo → lista de (vecino, peso). Se asume peso ≥ 0.
        source: nodo desde el cual calcular las distancias.

    Returns:
        Un par (dist, prev) donde dist[nodo] es la distancia mínima desde
        `source` y prev[nodo] el predecesor en el camino óptimo.
    """
    dist: Dict[Any, float] = {v: float('inf') for v in graph}
    prev: Dict[Any, Optional[Any]] = {v: None for v in graph}
    dist[source] = 0.0
    queue: List[Tuple[float, Any]] = [(0.0, source)]
    visited = set()
    while queue:
        d, u = heapq.heappop(queue)
        if u in visited:
            continue
        visited.add(u)
        for v, w in graph.get(u, []):
            if w < 0:
                raise ValueError("Dijkstra no admite pesos negativos")
            alt = d + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(queue, (alt, v))
    return dist, prev


def topological_sort(graph: Dict[Any, List[Any]]) -> List[Any]:
    """Calcula un orden topológico de un grafo dirigido acíclico.

    Args:
        graph: diccionario nodo → lista de vecinos (sin pesos).

    Returns:
        Lista de nodos en orden topológico. Lanza un ValueError si se detecta un ciclo.
    """
    from collections import deque

    in_deg = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_deg[v] = in_deg.get(v, 0) + 1
    queue = deque([u for u, deg in in_deg.items() if deg == 0])
    order: List[Any] = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph.get(u, []):
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    if len(order) != len(in_deg):
        raise ValueError("El grafo contiene ciclos; no se puede ordenar topológicamente")
    return order


def eigenvector_centrality(graph: Dict[Any, List[Tuple[Any, float]]], max_iter: int = 100, tol: float = 1e-6) -> Dict[Any, float]:
    """Calcula la centralidad por eigenvector de los nodos usando el método de la potencia.

    Args:
        graph: diccionario nodo → lista de (vecino, peso).
        max_iter: número máximo de iteraciones para el método de la potencia.
        tol: tolerancia para la convergencia.

    Returns:
        Diccionario nodo → valor de centralidad normalizado.
    """
    A, idx = adjacency_matrix(graph)
    n = A.shape[0]
    b_k = np.ones(n, dtype=float)
    b_k /= np.linalg.norm(b_k)
    for _ in range(max_iter):
        b_k1 = A @ b_k
        norm = np.linalg.norm(b_k1)
        if norm == 0:
            break
        b_k1 /= norm
        if np.linalg.norm(b_k1 - b_k) < tol:
            b_k = b_k1
            break
        b_k = b_k1
    # asignar resultados normalizados por suma
    centrality: Dict[Any, float] = {}
    total = np.sum(b_k)
    for node, i in idx.items():
        centrality[node] = float(b_k[i] / total if total != 0 else 0.0)
    return centrality