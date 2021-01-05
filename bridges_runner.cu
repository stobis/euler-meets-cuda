#include <iostream>
#include <unistd.h>
#include <unordered_map>

#include <moderngpu/context.hxx>
#include <moderngpu/memory.hxx>

#include "bridges.h"
#include "bridges_cpu.h"
#include "graph.hpp"
#include "graph_stats.hpp"

using namespace emc;
using namespace std;

// Misc
const string help_msg = "Command line arguments\n \
  -h -- print this help and exit\n \
  -i $input -- sets input file to $input\n \
  -o $output -- sets output file to $output\n \
  -a -- chooses algorithm to run:\n \
     naive - BFS + naive\n \
     tarjan - ECL-CC + Euler\n \
     hybrid - ECL-CC + naive\n \
     cpu - dfs on cpu\n \
     multi - CK on multicore cpu\n \
  -s -- print stats of the input graph and exit\n";

// Configuration of implementations to choose
typedef void (*BridgesFunction)(int, int, int const *, int const *, bool *, mgpu::context_t &);

std::unordered_map<std::string, BridgesFunction> bridges_algorithms = {
    {"naive", &cuda_bridges_naive}, {"tarjan", &cuda_bridges_tarjan}, {"hybrid", &cuda_bridges_hybrid}};

void print_ans(int const M, bool *is_bridge, bool verbose);

int main(int argc, char *argv[]) {
  std::ios_base::sync_with_stdio(false);

  char opt;
  char *input_file = NULL;
  char *output_file = NULL;
  BridgesFunction bridges_algorithm_to_run = NULL;
  bool run_cpu_alg = false;
  string alg;
  bool print_graph_stats = false;

  // Print help
  if (argc == 1) {
    cerr << help_msg << endl;
    exit(0);
  }

  // Parse args
  while ((opt = getopt(argc, argv, "hbi:o:a:s")) != EOF) {
    switch (opt) {
    case 'h':
      cerr << help_msg << endl;
      exit(0);

    case 'i':
      input_file = optarg;
      break;

    case 'o':
      output_file = optarg;
      break;

    case 'a': {
      alg = optarg;
      if (bridges_algorithms.count(alg) > 0)
        bridges_algorithm_to_run = bridges_algorithms[alg];
      else if (alg == "cpu" || alg == "multi") {
        run_cpu_alg = true;
      } else {
        cerr << "Unrecognized algorithm to use\n";
        exit(1);
      }
      break;
    }

    case 's':
      print_graph_stats = true;
    }
  }

  if (bridges_algorithm_to_run == NULL && run_cpu_alg == false && print_graph_stats == false) {
    cerr << help_msg << endl;
    cerr << "No algorithm specified. Exiting." << endl;
    exit(1);
  }

  // Read input
  Graph input_graph;
  if (input_file == NULL) {
    input_graph = Graph::read_from_stdin();
  } else {
    input_graph = Graph::read_from_file(input_file);
  }

  if (print_graph_stats) {
    print_stats(input_graph);
    return 0;
  }

  // Print output parser header
  std::cout << "%%% Bridges: File: " << (input_file == NULL ? std::string{"stdin"} : input_file) << std::endl;
  std::cout << "%%% N: " << input_graph.get_N() << std::endl;
  std::cout << "%%% M: " << input_graph.get_M() << std::endl;

  // Solve Bridges
  int const N = input_graph.get_N();
  int const M = input_graph.get_M();

  bool *is_bridge_host = new bool[M];

  if (run_cpu_alg) {
    if (alg == "cpu")
      cpu_bridges(N, M, input_graph.get_row_offsets().data(), input_graph.get_col_indices().data(),
                  is_bridge_host);
    else  // alg == "multi"
      multicore_cpu_bridges(
          N, M, input_graph.get_row_offsets().data(), input_graph.get_col_indices().data(), is_bridge_host);

  } else {
    mgpu::standard_context_t context(false);

    mgpu::mem_t<int> row_offsets = mgpu::to_mem(input_graph.get_row_offsets(), context);
    mgpu::mem_t<int> col_indices = mgpu::to_mem(input_graph.get_col_indices(), context);
    mgpu::mem_t<bool> is_bridge = mgpu::fill<bool>(false, M, context);

    (bridges_algorithm_to_run)(N, M, row_offsets.data(), col_indices.data(), is_bridge.data(), context);

    mgpu::dtoh(is_bridge_host, is_bridge.data(), M);
  }

  // Print Output
  print_ans(M, is_bridge_host, false);

  std::cout<<endl;

  delete[] is_bridge_host;

  return 0;
}

// Helpers
void print_ans(int const M, bool *ib, bool verbose) {
  int amount = 0;

  for (int i = 0; i < M; ++i) {
    if (verbose)
      std::cout << ib[i] << " ";
    if (ib[i])
      amount++;
  }
  if (verbose)
    std::cout << std::endl;
  std::cout << "%%% # Bridges: " << amount << std::endl;
}
