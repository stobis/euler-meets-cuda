#include "graph.hpp"
#include <iostream>

Graph::Graph(std::ifstream& in) {
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    in.read(reinterpret_cast<char*>(&m), sizeof(m));

    edges.resize(m);
    in.read(reinterpret_cast<char*>(edges.data()),
            m * sizeof(std::pair<int, int>));
}

Graph::Graph(int n, int m, std::vector<std::pair<int, int>> edges)
    : n(n), m(m), edges(edges) {}

int Graph::get_N() const { return n; }

int Graph::get_M() const { return m; }

// int Graph::get_M() { return m; }

std::vector<std::pair<int, int>> const& Graph::get_Edges() const {
    return edges;
}

Graph Graph::read_from_file(const char* filename) {
    std::ifstream in(filename, std::ios::binary);
    return Graph(in);
}

Graph Graph::read_from_stdin() {
    int n, m;
    std::vector<std::pair<int, int>> edges;

    std::cin >> n >> m;
    for (int i = 0; i < m; ++i) {
        int a, b;
        std::cin >> a >> b;
        edges.push_back(std::make_pair(a, b));
    }
    return Graph(n, m, edges);
}
