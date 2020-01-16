#include <iostream>
#include <unistd.h>

#include "lca/cuda_algorithms.h"
#include "lca/commons.h"
#include "lca/cuda_commons.h"

#include <moderngpu/context.hxx>
#include <moderngpu/memory.hxx>

using namespace std;

// Command line arguments
//  -b -- sets input to binary
//    -i input -- sets input file to $input
//    -o output -- sets output file to $output
//  -a [cuda-inlabel, ...] -- chooses algorithm to run
//

const char *help_msg = "Command line arguments\n \
  -b -- sets input to binary\n \
  -i $input -- sets input file to $input\n \
  -o $output -- sets output file to $output\n \
  -a [cuda-inlabel, ...] -- chooses algorithm to run\n";

enum Algorithm
{
  unspecified,
  cuda_inlabel_algorithm,
  cuda_naive_algorithm
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

  TestCase tc = is_binary_input ? readFromFile(input_file) : readFromStdIn();

  if (is_verbose)
    cerr << "N: " << tc.tree.V << endl
         << "Q: " << tc.q.N << endl
         << "Fathers size: " << tc.tree.father.size() << endl
         << "Qs size: " << tc.q.tab.size() << endl;

  vector<int> answers(tc.q.N);

  solve_lca(tc, algorithm, answers);

  if (output_file != NULL)
  {
    writeAnswersToFile(tc.q.N, answers.data(), output_file);
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
    cuda_lca_naive(tc.tree.V, cuda_queries, tc.q.N, cuda_parents, cuda_answers, context);

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

// int *dummy;
// mgpu::standard_context_t context(false);

// // run lca
// // int const N = 5;
// // int const root = 0;
// // std::vector<int> parents{-1, 0, 1, 0, 1};
// // int const Q = 5;
// // std::vector<int> queries{0, 2, 1, 4, 3, 2, 1, 2, 0, 4};

// int const N = 20;
// int const root = 11;
// std::vector<int> parents{18, 12, 19, 9, 13, 1, 8, 18, 14, 12, 14, -1, 11, 11, 16, 18, 11, 15, 16, 11};
// int const Q = 10;
// std::vector<int> queries{0, 13, 3, 19, 1, 1, 11, 17, 15, 7, 2, 17, 0, 10, 6, 1, 3, 2, 0, 13};

// mgpu::mem_t<int> cuda_parents = mgpu::to_mem(parents, context);
// mgpu::mem_t<int> cuda_queries = mgpu::to_mem(queries, context);
// mgpu::mem_t<int> cuda_answers = mgpu::fill<int>(0, Q, context);

// // cuda_lca_naive( N, cuda_parents.data(), Q, cuda_queries.data(), cuda_answers.data(), context );
// cuda_lca_inlabel(N, cuda_parents.data(), Q, cuda_queries.data(), cuda_answers.data(), context);

// std::vector<int> correct_answers{11, 11, 1, 11, 18, 11, 16, 11, 11, 11};

// bool isOk = true;
// int *answers = new int[Q];
// mgpu::dtoh(answers, cuda_answers.data(), Q);
// for (int i = 0; i < Q; i++)
// {
//   if (answers[i] != correct_answers[i])
//     isOk = false;
// }

// if (isOk)
// {
//   std::cout << "LCA OK" << std::endl;
// }
// else
// {
//   std::cout << "LCA ERROR!" << std::endl;
//   for (int i = 0; i < Q; i++)
//   {
//     std::cout << answers[i] << " ";
//   }
//   std::cout << std::endl;
// }

// return 0;