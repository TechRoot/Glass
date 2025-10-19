// Algoritmo de Dijkstra para caminos mínimos en grafos ponderados sin pesos negativos.

#pragma once

#include <vector>
#include <utility>
#include <limits>

namespace graphs {

// Representación del grafo: vector de listas de pares (vecino, peso)
using AdjList = std::vector<std::vector<std::pair<int, double>>>;

// Calcula las distancias mínimas desde el nodo source en un grafo de n nodos.
// Devuelve un vector de distancias de tamaño n. Las distancias no alcanzadas
// quedan en infinity.
std::vector<double> dijkstra(int n, const AdjList &adj, int source);

} // namespace graphs