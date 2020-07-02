#include <iostream>
#include <unistd.h>
#include <unordered_map>

#include <moderngpu/context.hxx>
#include <moderngpu/memory.hxx>

#include "lca.h"
#include "tree.h"
#include "utils.h"

using namespace std;

const string help_msg = "Command line arguments\n \
  -h -- print this help and exit\n \
  -i $input -- sets input file to $input\n \
  -o $output -- sets output file to $output\n \
  -a [cuda-inlabel, cuda-naive, cpu-inlabel, cpu-rmq, cpu-simple] -- chooses algorithm to run\n";

typedef void (*CpuLcaFunction)(int N, const int *parents, int Q, const int *queries, int *answers);
typedef void (*CudaLcaFunction)(int N, const int *parents, int Q, const int *queries, int *answers,
                                int batchSize, mgpu::context_t &context);

unordered_map<string, CpuLcaFunction> cpu_algorithms = {
    {"cpu-inlabel", &cpu_lca_inlabel}, {"cpu-rmq", &cpu_lca_rmq}, {"cpu-simple", &cpu_lca_simple}};

unordered_map<string, CudaLcaFunction> cuda_algorithms = {{"cuda-inlabel", &cuda_lca_inlabel},
                                                          {"cuda-naive", &cuda_lca_naive}};

int main(int argc, char *argv[]) {
  std::ios_base::sync_with_stdio(false);

  char opt;
  char *input_file = NULL;
  char *output_file = NULL;
  CpuLcaFunction cpu_algorithm_to_use = NULL;
  CudaLcaFunction cuda_algorithm_to_use = NULL;
  int batchSize = -1;

  // Print help
  if (argc == 1) {
    cerr << help_msg << endl;
    exit(0);
  }

  // Parse args
  while ((opt = getopt(argc, argv, "hb:i:o:a:")) != EOF) {
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
      if (cpu_algorithms.count(alg) > 0)
        cpu_algorithm_to_use = cpu_algorithms[alg];
      else if (cuda_algorithms.count(alg) > 0)
        cuda_algorithm_to_use = cuda_algorithms[alg];
      else {
        cerr << "Unrecognized algorithm to use: " << optarg << "\n";
        exit(1);
      }
      break;
    }

    case 'b':
      batchSize = atoi(optarg);
      break;
    }
  }

  // Read input
  LcaTestCase tc = input_file == NULL ? readFromStdIn() : readFromFile(input_file);
  vector<int> answers(tc.q.N);

  // Default batch size
  if (batchSize == -1)
    batchSize = tc.q.N;

  // Print output parser header
  std::cout << "%%% Lca: File: " << (input_file == NULL ? std::string{"stdin"} : input_file) << std::endl;
  std::cout << "%%% N: " << tc.tree.V << std::endl;

  // Solve LCA
  if (cuda_algorithm_to_use != NULL) {
    mgpu::standard_context_t context(0);
    int *cuda_parents, *cuda_queries, *cuda_answers;

    CUCHECK(cudaMalloc((void **)&cuda_parents, sizeof(int) * tc.tree.V));
    CUCHECK(cudaMemcpy(cuda_parents, tc.tree.father.data(), sizeof(int) * tc.tree.V, cudaMemcpyHostToDevice));

    CUCHECK(cudaMalloc((void **)&cuda_queries, sizeof(int) * tc.q.N * 2));
    CUCHECK(cudaMemcpy(cuda_queries, tc.q.tab.data(), sizeof(int) * tc.q.N * 2, cudaMemcpyHostToDevice));

    CUCHECK(cudaMalloc((void **)&cuda_answers, sizeof(int) * tc.q.N));

    cuda_algorithm_to_use(tc.tree.V, cuda_parents, tc.q.N, cuda_queries, cuda_answers, batchSize, context);

    CUCHECK(cudaMemcpy(answers.data(), cuda_answers, sizeof(int) * tc.q.N, cudaMemcpyDeviceToHost));
  } else if (cpu_algorithm_to_use != NULL) {
    cpu_algorithm_to_use(tc.tree.V, tc.tree.father.data(), tc.q.N, tc.q.tab.data(), answers.data());
  } else {
    cerr << help_msg << endl;
    cerr << "No algorithm specified. Exiting." << endl;
    exit(1);
  }

  cout << "%%% numQ: " << tc.q.N << endl;

  pair<int, double> height = getHeight(tc.tree);
  cout << "%%% MaxHeight: " << height.first << endl;
  cout << "%%% AvgHeight: " << height.second << endl;

  // Print Output
  if (output_file != NULL) {
    writeAnswersToFile(tc.q.N, answers.data(), output_file);
  } else {
    writeAnswersToStdOut(tc.q.N, answers.data());
  }

  return 0;
}