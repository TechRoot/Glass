#!/usr/bin/env python3
"""
CLI para ejecutar algoritmos de grafos.

Permite calcular la matriz Laplaciana, el valor de Fiedler, caminos mínimos
con Dijkstra, orden topológico en DAG y centralidad por eigenvector a partir
de un grafo definido en un archivo JSON o CSV. El formato JSON esperado es
un diccionario donde cada clave es un nodo y su valor es una lista de
objetos con campos "to" y opcionalmente "weight". El formato CSV debe
contener tres columnas: origen,destino,peso. Si no se especifica un peso,
se asume 1.0.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from ..lib import graph_algorithms as ga


def configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(levelname)s: %(message)s")


def load_graph(file_path: str) -> Dict[str, List[Tuple[str, float]]]:
    """Carga un grafo desde un archivo JSON o CSV.

    Args:
        file_path: ruta al archivo.

    Returns:
        Diccionario nodo → lista de (vecino, peso).
    """
    path = Path(file_path)
    graph: Dict[str, List[Tuple[str, float]]] = {}
    if path.suffix.lower() in {'.json'}:
        with open(path) as f:
            data = json.load(f)
        for u, edges in data.items():
            graph[u] = []
            for edge in edges:
                v = edge['to']
                w = float(edge.get('weight', 1.0))
                graph[u].append((v, w))
    elif path.suffix.lower() in {'.csv'}:
        with open(path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                u = row[0]
                v = row[1] if len(row) > 1 else None
                w = float(row[2]) if len(row) > 2 else 1.0
                if u not in graph:
                    graph[u] = []
                if v is not None:
                    graph[u].append((v, w))
    else:
        raise ValueError(f"Tipo de archivo no soportado: {path.suffix}")
    return graph


def cmd_laplacian(args) -> None:
    graph = load_graph(args.input)
    L = ga.laplacian_matrix(graph)
    print("Matriz Laplaciana:")
    with np_print_options(precision=4, suppress=True):
        print(L)


def cmd_fiedler(args) -> None:
    graph = load_graph(args.input)
    value = ga.fiedler_value(graph)
    print(f"Valor de Fiedler: {value:.6f}")


def cmd_dijkstra(args) -> None:
    graph = load_graph(args.input)
    source = args.source
    if source not in graph:
        logging.error(f"El nodo origen '{source}' no existe en el grafo")
        return
    distances, prev = ga.dijkstra(graph, source)
    for node, dist in distances.items():
        print(f"{source} -> {node}: distancia = {dist:.4f}")


def cmd_topo(args) -> None:
    graph = load_graph(args.input)
    # Convertir a representación sin pesos para topológico
    graph_simple = {u: [v for v, _ in edges] for u, edges in graph.items()}
    try:
        order = ga.topological_sort(graph_simple)
        print("Orden topológico:", ", ".join(str(x) for x in order))
    except ValueError as e:
        print(f"Error: {e}")


def cmd_centrality(args) -> None:
    graph = load_graph(args.input)
    centrality = ga.eigenvector_centrality(graph)
    for node, value in centrality.items():
        print(f"Centralidad {node}: {value:.6f}")


class np_print_options:
    """Context manager para configurar numpy printoptions temporalmente."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.orig = None
    def __enter__(self):
        import numpy as _np
        self.orig = _np.get_printoptions()
        _np.set_printoptions(**self.kwargs)
    def __exit__(self, exc_type, exc, exc_tb):
        import numpy as _np
        _np.set_printoptions(**self.orig)


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI para análisis de grafos")
    parser.add_argument('--dry-run', action='store_true', default=True, help='Simula operaciones sin escribir archivos')
    parser.add_argument('--confirm', action='store_true', help='Confirma operaciones que generan salidas (no usado)')
    parser.add_argument('--log-level', default='INFO', help='Nivel de logging')
    subparsers = parser.add_subparsers(dest='command', required=True)
    # Laplacian
    p_lap = subparsers.add_parser('laplacian', help='Calcula la matriz Laplaciana de un grafo')
    p_lap.add_argument('--input', required=True, help='Archivo JSON o CSV con la definición del grafo')
    p_lap.set_defaults(func=cmd_laplacian)
    # Fiedler
    p_fie = subparsers.add_parser('fiedler', help='Calcula el valor de Fiedler de un grafo')
    p_fie.add_argument('--input', required=True, help='Archivo JSON o CSV con la definición del grafo')
    p_fie.set_defaults(func=cmd_fiedler)
    # Dijkstra
    p_dij = subparsers.add_parser('dijkstra', help='Calcula las distancias mínimas desde un nodo')
    p_dij.add_argument('--input', required=True, help='Archivo JSON o CSV con la definición del grafo')
    p_dij.add_argument('--source', required=True, help='Nodo origen')
    p_dij.set_defaults(func=cmd_dijkstra)
    # Topological sort
    p_topo = subparsers.add_parser('topo', help='Ordenación topológica de un DAG')
    p_topo.add_argument('--input', required=True, help='Archivo JSON o CSV con la definición del grafo')
    p_topo.set_defaults(func=cmd_topo)
    # Eigenvector centrality
    p_cen = subparsers.add_parser('centrality', help='Calcula la centralidad por eigenvector')
    p_cen.add_argument('--input', required=True, help='Archivo JSON o CSV con la definición del grafo')
    p_cen.set_defaults(func=cmd_centrality)
    args = parser.parse_args()
    configure_logging(args.log_level)
    args.func(args)


if __name__ == '__main__':
    main()