#include "dijkstra.hpp"

#include <cassert>
#include <cmath>

int main() {
    // Construcción de un grafo dirigido ponderado
    // 0 -> 1 (1), 0 -> 2 (4), 1 -> 2 (2), 1 -> 3 (5), 2 -> 3 (1)
    graphs::AdjList adj(4);
    adj[0].push_back({1, 1.0});
    adj[0].push_back({2, 4.0});
    adj[1].push_back({2, 2.0});
    adj[1].push_back({3, 5.0});
    adj[2].push_back({3, 1.0});
    // ejecutar Dijkstra desde 0
    auto dist = graphs::dijkstra(4, adj, 0);
    // La distancia de 0 a 3 es 4 (0->1->2->3)
    assert(std::fabs(dist[3] - 4.0) < 1e-9);
    // La distancia a 2 es 3 (0->1->2)
    assert(std::fabs(dist[2] - 3.0) < 1e-9);
    // La distancia a 1 es 1
    assert(std::fabs(dist[1] - 1.0) < 1e-9);
    return 0;
}