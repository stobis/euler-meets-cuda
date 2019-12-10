#include "bridges/cpu-bridges-dfs.hpp"

#include <map>
#include <vector>

#include "graph.hpp"
#include "test-result.hpp"
#include "timer.hpp"

int visit_time;
std::vector<std::vector<int> > G;
std::map<std::pair<int, int>, bool> bridges;

struct node {
    int preorder, low, parent;
    bool visited;

    node(int parent = -1, int visit_time = 0)
        : preorder(visit_time), low(preorder), visited(visit_time > 0), parent(parent) {}
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

    // if (nodes[start].low == nodes[start].preorder && parent != -1) {
    //     bridges[std::make_pair(std::min(start, parent),
    //                            std::max(start, parent))] = true;
    // }
    return;
}

void prepare(int n) {
    G.resize(n + 1);
    nodes.resize(n + 1);
    bridges.clear();
    return;
}

TestResult sequential_dfs(Graph const& graph) {
    Timer timer("cpu-dfs");

    prepare(graph.get_N());

    auto const edges = graph.get_Edges();

    for (auto const& e : edges) {
        G[e.first].push_back(e.second);
        G[e.second].push_back(e.first);
    }
    
    visit_time = 1;
    dfs(1, -1);

    TestResult result(graph.get_M());
    int it = 0;
    for (auto const& e : edges) {
        // auto sorted_pair = e;
        // if (sorted_pair.first > sorted_pair.second)
        //     std::swap(sorted_pair.first, sorted_pair.second);
        // if (bridges[sorted_pair]) {
        //     result[it] = 1;
        // }
        // it++;
        int act = e.first, act_p = e.second;
        if (nodes[act].parent == act_p) {
            // tree edge
            if (nodes[act].low == nodes[act].preorder) {
                result[it] = 1;
            }
        }
        std::swap(act, act_p);
        if (nodes[act].parent == act_p) {
            // tree edge
            if (nodes[act].low == nodes[act].preorder) {
                result[it] = 1;
            }
        }
        it++;
    }

    timer.stop();
    timer.print_info("Overall");

    return result;
}
