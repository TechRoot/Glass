"""
Interfaz de línea de comandos para algoritmos de grafos.

Este script expone varias operaciones sobre grafos a través de la línea de comandos:
calcula la matriz laplaciana y el valor de Fiedler a partir de una lista de aristas,
calcula caminos mínimos desde una única fuente, realiza ordenación topológica
en DAG y calcula la centralidad de vector propio.  La entrada puede
proporcionarse como archivos JSON que mapean nodos a listas de adyacencia
o archivos CSV de listas de aristas【293886591685374†L68-L155】.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..lib.graph_algorithms import adjacency_matrix, laplacian_matrix, fiedler_value, dijkstra, topological_sort, eigenvector_centrality


def load_edges(path: Path) -> Tuple[List[Tuple[int, int]], int]:
    edges: List[Tuple[int, int]] = []
    max_node = -1
    with path.open(newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            u, v = int(row[0]), int(row[1])
            edges.append((u, v))
            max_node = max(max_node, u, v)
    return edges, max_node + 1


def load_adj_json(path: Path) -> Dict[int, List[int]]:
    with path.open() as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI de grafos: algoritmos básicos")
    sub = parser.add_subparsers(dest='command', required=True)

    lap = sub.add_parser('laplacian', help='Calcular la laplaciana y el valor de Fiedler a partir de un CSV de aristas')
    lap.add_argument('edges', type=Path, help='CSV con aristas (u,v)')

    dij = sub.add_parser('dijkstra', help='Caminos mínimos desde una única fuente a partir de un JSON de adyacencia')
    dij.add_argument('adj', type=Path, help='JSON con lista de adyacencia {nodo: [[vecino, peso], ...]}')
    dij.add_argument('source', type=int, help='Nodo de origen')

    topo = sub.add_parser('toposort', help='Ordenación topológica de un DAG a partir de un JSON de adyacencia')
    topo.add_argument('adj', type=Path, help='JSON con lista de adyacencia {nodo: [vecinos]}')

    eig = sub.add_parser('eigenvec', help='Centralidad de vector propio a partir de un JSON de adyacencia')
    eig.add_argument('adj', type=Path, help='JSON con lista de adyacencia {nodo: [vecinos]}')

    args = parser.parse_args()
    if args.command == 'laplacian':
        edges, n = load_edges(args.edges)
        A = adjacency_matrix(edges, n)
        L = laplacian_matrix(A)
        print("Laplaciana:\n", L)
        print("Valor de Fiedler:", fiedler_value(L))
    elif args.command == 'dijkstra':
        with args.adj.open() as f:
            adj = json.load(f)
        adj = {int(k): [(int(v[0]), float(v[1])) for v in vals] for k, vals in adj.items()}
        dist = dijkstra(adj, args.source)
        print(dist)
    elif args.command == 'toposort':
        adj = load_adj_json(args.adj)
        order = topological_sort(adj)
        print(order)
    elif args.command == 'eigenvec':
        adj = load_adj_json(args.adj)
        centrality = eigenvector_centrality(adj)
        print(centrality)


if __name__ == '__main__':
    main()
