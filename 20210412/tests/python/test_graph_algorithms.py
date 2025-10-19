import numpy as np

from ...lib import graph_algorithms as ga


def test_laplacian_and_fiedler():
    # Grafo de estrella de 3 nodos: 0 conectado a 1 y 2
    graph = {
        0: [(1, 1.0), (2, 1.0)],
        1: [(0, 1.0)],
        2: [(0, 1.0)]
    }
    L = ga.laplacian_matrix(graph)
    expected = np.array([[ 2, -1, -1],
                         [-1,  1,  0],
                         [-1,  0,  1]], dtype=float)
    assert np.allclose(L, expected)
    # Valores propios: 0, 1, 3 -> Fiedler = 1
    fiedler = ga.fiedler_value(graph)
    assert abs(fiedler - 1.0) < 1e-5


def test_dijkstra_simple_graph():
    # Grafo dirigido con pesos no negativos
    graph = {
        'A': [('B', 1.0), ('C', 4.0)],
        'B': [('C', 2.0), ('D', 5.0)],
        'C': [('D', 1.0)],
        'D': []
    }
    dist, prev = ga.dijkstra(graph, 'A')
    assert abs(dist['D'] - 4.0) < 1e-9  # A->C->D = 4
    assert prev['D'] == 'C'
    assert prev['C'] == 'B' or prev['C'] == 'A'  # C puede venir de B o A dependiendo del orden


def test_topological_sort_validity():
    # Grafo acíclico dirigido
    graph = {
        'p': [('q', 1.0), ('r', 1.0)],
        'q': [('s', 1.0)],
        'r': [('s', 1.0)],
        's': []
    }
    # Para topológico ignoramos pesos
    graph_simple = {u: [v for v, _ in edges] for u, edges in graph.items()}
    order = ga.topological_sort(graph_simple)
    # comprobar que para cada arista u->v, u aparece antes que v en el orden
    pos = {node: i for i, node in enumerate(order)}
    for u in graph_simple:
        for v in graph_simple[u]:
            assert pos[u] < pos[v]


def test_eigenvector_centrality_symmetry():
    # Grafo no dirigido en ciclo de 3 nodos
    graph = {
        0: [(1, 1.0), (2, 1.0)],
        1: [(0, 1.0), (2, 1.0)],
        2: [(0, 1.0), (1, 1.0)]
    }
    centrality = ga.eigenvector_centrality(graph)
    # Todos los nodos deben tener centralidad igual en un ciclo completo
    vals = list(centrality.values())
    assert np.allclose(vals, vals[0], atol=1e-6)