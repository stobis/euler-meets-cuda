#include <iostream>
#include <unistd.h>
#include <unordered_map>

#include <moderngpu/context.hxx>
#include <moderngpu/memory.hxx>

#include "bridges.h"
#include "graph.hpp"

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
     hybrid - ECL-CC + naive\n";

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

  // Print help
  if (argc == 1) {
    cerr << help_msg << endl;
    exit(0);
  }

  // Parse args
  while ((opt = getopt(argc, argv, "hbi:o:a:")) != EOF) {
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
      string alg = optarg;
      if (bridges_algorithms.count(alg) > 0)
        bridges_algorithm_to_run = bridges_algorithms[alg];
      else {
        cerr << "Unrecognized algorithm to use\n";
        exit(1);
      }
      break;
    }
    }
  }

  if (bridges_algorithm_to_run == NULL) {
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

  // Print output parser header
  std::cout << "%%% Bridges: File: " << (input_file == NULL ? std::string{"stdin"} : input_file) << std::endl;
  std::cout << "%%% N: " << input_graph.get_N() << std::endl;
  std::cout << "%%% M: " << input_graph.get_M() << std::endl;

  // Solve Bridges
  mgpu::standard_context_t context(false);

  int const N = input_graph.get_N();
  int const M = input_graph.get_M();
  mgpu::mem_t<int> row_offsets = mgpu::to_mem(input_graph.get_row_offsets(), context);
  mgpu::mem_t<int> col_indices = mgpu::to_mem(input_graph.get_col_indices(), context);
  mgpu::mem_t<bool> is_bridge = mgpu::fill<bool>(false, M, context);

  (bridges_algorithm_to_run)(N, M, row_offsets.data(), col_indices.data(), is_bridge.data(), context);

  // Print Output
  print_ans(M, is_bridge.data(), false);

  return 0;
}

// Helpers
void print_ans(int const M, bool *is_bridge, bool verbose) {
  int amount = 0;
  bool *ib = new bool[M];
  mgpu::dtoh(ib, is_bridge, M);
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