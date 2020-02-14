#include <iostream>
#include <unordered_map>

#include "bridges/graph.hpp"
#include "bridges/cuda_algorithms.h"

#include <moderngpu/context.hxx>
#include <moderngpu/memory.hxx>

// Misc
const std::string _usage =
    " VARIANT IN\n\n"
    "Available variants:\n"
    "gpu-bfs - simple parallel BFS implementation (naive)\n"
    "gpu-tarjan - parallel implementation using fast CC and Euler Tour technique\n";
    // "gpu-hybrid - parallel implementation combining spanning tree given by BFS with ET technique\n";

// Configuration of implementations to choose
typedef void (*BridgesFunction)(int, int, int const *, int const *, bool *, mgpu::context_t &);

std::unordered_map<std::string, BridgesFunction> algorithms = {
    {"gpu-bfs", &cuda_bridges_naive},
    {"gpu-tarjan", &cuda_bridges_tarjan}
};

// Helpers
void print_ans(int const M, bool *is_bridge, bool verbose)
{
  int amount = 0;
  bool *ib = new bool[M];
  mgpu::dtoh(ib, is_bridge, M);
  for (int i = 0; i < M; ++i) {
      if (verbose) std::cout << ib[i] << " ";
      if (ib[i]) amount++;
  }
  if (verbose) std::cout << std::endl;
  std::cout << "%%% Bridges: " << amount << std::endl;
}

// Main func
int main(int argc, char *argv[])
{
    std::ios_base::sync_with_stdio(false);

    // Parse args
    if (argc < 2 || !algorithms.count(argv[1])) {
        std::cerr << "Usage: " << argv[0] << _usage << std::endl;
        exit(1);
    }

    std::vector<std::string> args(argv, argv + argc);

    // Read input
    Graph input_graph;
    if (argc < 3) {
      input_graph = Graph::read_from_stdin();
    } else {
      input_graph = Graph::read_from_file(argv[2]);
    }

    std::cout << "%%% File: " << (argc > 2 ? args[2] : std::string{"stdin"}) << std::endl;
    std::cout << "%%% N: " << input_graph.get_N() << std::endl;
    std::cout << "%%% M: " << input_graph.get_M() << std::endl; 

    // Exec
    mgpu::standard_context_t context(false);

    int const N = input_graph.get_N();
    int const M = input_graph.get_M();
    mgpu::mem_t<int> row_offsets = mgpu::to_mem(input_graph.get_row_offsets(), context);
    mgpu::mem_t<int> col_indices = mgpu::to_mem(input_graph.get_col_indices(), context);
    mgpu::mem_t<bool> is_bridge = mgpu::fill<bool>(false, M, context);

    (*algorithms[args[1]])(N, M, row_offsets.data(), col_indices.data(), is_bridge.data(), context);

    print_ans(M, is_bridge.data(), false);
    
    return 0;
}
