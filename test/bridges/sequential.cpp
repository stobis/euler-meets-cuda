#include <iostream>
#include <map>
#include <vector>

#include "graph.hpp"
#include "timer.hpp"

int visit_time;
std::vector<std::vector<int>> G;
std::vector<bool> ans;

struct node {
    int preorder, low, parent;
    bool visited;

    node(int parent = -1, int visit_time = 0)
        : preorder(visit_time),
          low(preorder),
          visited(visit_time > 0),
          parent(parent) {}
};

std::vector<node> nodes;

void dfs(int start, int parent) {
    nodes[start] = node(parent, visit_time++);

    for (auto x : G[start]) {
        if (x == parent) continue;
        if (!nodes[x].visited) {
            dfs(x, start);

            nodes[start].low = std::min(nodes[start].low, nodes[x].low);
        } else {
            nodes[start].low = std::min(
                nodes[start].low, nodes[x].preorder);  // non dfs-tree edge
        }
    }
}

bool is_bridge(int x, int y) {
    if (nodes[x].parent == y) {
        // tree edge
        if (nodes[x].low == nodes[x].preorder) {
            return true;
        }
    }
    return false;
}

void print_ans(std::vector<bool> const & ans, bool verbose) {
    int amount = 0;
    for (auto x : ans) {
        if (verbose) std::cout << x << " ";
        if (x) amount++;
    }
    if (verbose) std::cout << std::endl;
    std::cout << "%%% Bridges: " << amount << std::endl;
}

int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);

    // Parse args
    // if (argc < 2) {
    //     std::cerr << "Usage: " << argv[0] << " IN" << std::endl;
    //     exit(1);
    // }

    // Read input
    Graph input_graph;
    if (argc < 2) {
        input_graph = Graph::read_from_stdin();
    } else {
        input_graph = Graph::read_from_file(argv[1]);
    }

    std::cout << "%%% File: "
              << (argc > 1 ? std::string{argv[1]} : std::string{"stdin"})
              << std::endl;
    std::cout << "%%% N: " << input_graph.get_N() << std::endl;
    std::cout << "%%% M: " << input_graph.get_M() << std::endl;

    int const N = input_graph.get_N();
    int const M = input_graph.get_M();
    auto const& row_offsets = input_graph.get_row_offsets();
    auto const& col_indices = input_graph.get_col_indices();

    // Prepare
    ans.resize(M);
    nodes.resize(N);
    G.resize(N);
    for (int i = 0; i < N; ++i) {
        G[i].clear();
        for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j) {
            G[i].push_back(col_indices[j]);
            G[col_indices[j]].push_back(i);
        }
    }

    // Run
    Timer timer("cpu-dfs");

    visit_time = 1;
    dfs(0, -1);

    for (int i = 0; i < N; ++i) {
        for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j) {
            int x = i, y = col_indices[j];
            if (is_bridge(x, y) || is_bridge(y, x)) {
                ans[j] = true;
            }
        }
    }

    timer.stop();
    timer.print_info("Overall");

    // Print ans
    print_ans(ans, false);

    return 0;
}
