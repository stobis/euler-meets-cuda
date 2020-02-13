#include <iostream>
#include <unistd.h>

#include "lca/cuda_algorithms.h"
#include "lca/commons.h"
#include "lca/cuda_commons.h"
#include "lca/cpu_lca_inlabel.h"
#include "lca/cpu_lca_rmq.h"
#include "lca/cpu_lca_simple.h"

#include <moderngpu/context.hxx>
#include <moderngpu/memory.hxx>

using namespace std;

const char *help_msg = "Command line arguments\n \
  -b -- sets input to binary\n \
  -i $input -- sets input file to $input\n \
  -o $output -- sets output file to $output\n \
  -a [cuda-inlabel, cuda-naive, cpu-inlabel, cpu-rmq, cpu-simple] -- chooses algorithm to run\n";

enum Algorithm
{
  unspecified,
  cuda_inlabel_algorithm,
  cuda_naive_algorithm,
  cpu_inlabel_algorithm,
  cpu_rmq_algorithm,
  cpu_simple_algorithm
};
bool is_cuda_algorithm(Algorithm alg);

void solve_lca(TestCase &tc, Algorithm algorithm, vector<int> &answers);

int main(int argc, char *argv[])
{
  std::ios_base::sync_with_stdio(false);

  char opt;
  bool is_binary_input = false;
  char *input_file = NULL, *output_file = NULL;
  Algorithm algorithm = unspecified;
  bool is_verbose = false;

  while ((opt = getopt(argc, argv, "hbi:o:a:v")) != EOF)
  {
    switch (opt)
    {
    case 'h':
      cerr << help_msg << endl;
      exit(0);

    case 'b':
      is_binary_input = true;
      break;

    case 'i':
      input_file = optarg;
      break;

    case 'o':
      output_file = optarg;
      break;

    case 'a':
      if (strcmp(optarg, "cuda-inlabel") == 0)
        algorithm = cuda_inlabel_algorithm;
      else if (strcmp(optarg, "cuda-naive") == 0)
        algorithm = cuda_naive_algorithm;
      else if (strcmp(optarg, "cpu-inlabel") == 0)
        algorithm = cpu_inlabel_algorithm;
      else if (strcmp(optarg, "cpu-rmq") == 0)
        algorithm = cpu_rmq_algorithm;
      else if (strcmp(optarg, "cpu-simple") == 0)
        algorithm = cpu_simple_algorithm;
      else
      {
        cerr << "Unrecognized algorithm to use\n";
        exit(1);
      }
      break;

    case 'v':
      is_verbose = true;
      break;
    }
  }

  if (algorithm == unspecified)
  {
    cerr << "No algorithm specified, using cuda-inlabel" << endl;
    algorithm = cuda_inlabel_algorithm;
  }

  TestCase tc = input_file == NULL ? readFromStdIn() : readFromFile(is_binary_input, input_file);

  if (is_verbose)
    cerr << "N: " << tc.tree.V << endl
         << "Q: " << tc.q.N << endl
         << "Fathers size: " << tc.tree.father.size() << endl
         << "Qs size: " << tc.q.tab.size() << endl;

  vector<int> answers(tc.q.N);

  solve_lca(tc, algorithm, answers);

  if (output_file != NULL)
  {
    writeAnswersToFile(is_binary_input, tc.q.N, answers.data(), output_file);
  }
  else
  {
    writeAnswersToStdOut(tc.q.N, answers.data());
  }

  return 0;
}

void solve_lca(TestCase &tc, Algorithm algorithm, vector<int> &answers)
{
  mgpu::standard_context_t context(0);

  int *cuda_parents, *cuda_queries, *cuda_answers;

  if (is_cuda_algorithm(algorithm))
  {
    CUCHECK(cudaMalloc((void **)&cuda_parents, sizeof(int) * tc.tree.V));
    CUCHECK(cudaMemcpy(cuda_parents, tc.tree.father.data(), sizeof(int) * tc.tree.V, cudaMemcpyHostToDevice));

    CUCHECK(cudaMalloc((void **)&cuda_queries, sizeof(int) * tc.q.N * 2));
    CUCHECK(cudaMemcpy(cuda_queries, tc.q.tab.data(), sizeof(int) * tc.q.N * 2, cudaMemcpyHostToDevice));

    CUCHECK(cudaMalloc((void **)&cuda_answers, sizeof(int) * tc.q.N));
  }

  switch (algorithm)
  {
  case cuda_inlabel_algorithm:
    cuda_lca_inlabel(tc.tree.V, cuda_parents, tc.q.N, cuda_queries, cuda_answers, context);
    break;
  case cuda_naive_algorithm:
    cuda_lca_naive(tc.tree.V, cuda_parents, tc.q.N, cuda_queries, cuda_answers, context);
    break;
  case cpu_inlabel_algorithm:
    cpu_lca_inlabel(tc.tree.V, tc.tree.father.data(), tc.q.N, tc.q.tab.data(), answers.data());
    break;
  case cpu_rmq_algorithm:
    cpu_lca_rmq(tc.tree.V, tc.tree.father.data(), tc.q.N, tc.q.tab.data(), answers.data());
    break;
  case cpu_simple_algorithm:
    cpu_lca_simple(tc.tree.V, tc.tree.father.data(), tc.q.N, tc.q.tab.data(), answers.data());
    break;

  default:
    //Should not happen
    cerr << "Bad algorithm specified" << endl;
    exit(1);
    break;
  }

  if (is_cuda_algorithm(algorithm))
  {
    CUCHECK(cudaMemcpy(answers.data(), cuda_answers, sizeof(int) * tc.q.N, cudaMemcpyDeviceToHost));
  }
}

bool is_cuda_algorithm(Algorithm alg)
{
  switch (alg)
  {
  case cuda_inlabel_algorithm:
  case cuda_naive_algorithm:
    return true;

  default:
    return false;
  }
}