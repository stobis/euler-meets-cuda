#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <utility>

typedef std::pair<int, int> pii;

typedef std::pair<int, int> edge;

class Graph {
   private:
    int n, m;
    std::vector<std::pair<int, int>> edges;

   public:
    Graph(std::ifstream&);
    Graph(int, int, std::vector<std::pair<int, int>>);

    int get_N() const;
    int get_M() const;
    // int get_M();
    std::vector<std::pair<int, int>> const& get_Edges() const;

    static Graph read_from_file(const char*);
    static Graph read_from_stdin();
};

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


// Main func
int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false);

    Graph const input_graph = Graph::read_from_file(argv[1]);
    // Graph input_graph = Graph::read_from_stdin();
    std::cout << "%%% File: " << argv[1] << std::endl;
    std::cout << "%%% N: " << input_graph.get_N() << std::endl;
    std::cout << "%%% M: " << input_graph.get_M() << std::endl; 
    // auto xd = input_graph.get_Edges(); 
    // for (auto e : xd) {
    //     std::cout << e.first << " " << e.second << std::endl;
    // }
    // parallel_cc(input_graph);
    // return 0;
    int const N = input_graph.get_N();
    int const M = input_graph.get_M();

    std::vector<int> labels(N);
    for (int i = 1; i <= N; ++i) labels[i-1] = i;
    std::random_shuffle(labels.begin(), labels.end());

    std::vector<std::pair<int, int>> edges = input_graph.get_Edges();

    for (edge & e : edges) {
        e.first = labels[e.first-1];
        e.second = labels[e.second-1];
    }

    std::ofstream out(argv[2], std::ios::binary);
    assert(out.is_open());

    out.write((char *)&N, sizeof(int));
    out.write((char *)&M, sizeof(int));
    out.write((char *)edges.data(), edges.size() * sizeof(pii));

    return 0;
}
