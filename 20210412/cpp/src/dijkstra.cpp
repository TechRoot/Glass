#include "dijkstra.hpp"

#include <queue>

namespace graphs {

struct NodeCmp {
    bool operator()(const std::pair<double, int> &a, const std::pair<double, int> &b) const {
        return a.first > b.first;
    }
};

std::vector<double> dijkstra(int n, const AdjList &adj, int source) {
    std::vector<double> dist(n, std::numeric_limits<double>::infinity());
    std::vector<bool> visited(n, false);
    dist[source] = 0.0;
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, NodeCmp> pq;
    pq.emplace(0.0, source);
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (visited[u]) {
            continue;
        }
        visited[u] = true;
        for (const auto &edge : adj[u]) {
            int v = edge.first;
            double w = edge.second;
            if (w < 0.0) {
                throw std::runtime_error("Dijkstra no admite pesos negativos");
            }
            double alt = d + w;
            if (alt < dist[v]) {
                dist[v] = alt;
                pq.emplace(alt, v);
            }
        }
    }
    return dist;
}

} // namespace graphs