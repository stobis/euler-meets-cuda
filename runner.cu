#include <iostream>

#include "bridges/cuda_algorithms.h"
#include "lca/cuda_algorithms.h"

#include <moderngpu/context.hxx>
#include <moderngpu/memory.hxx>

int main(int argc, char *argv[])
{
  std::ios_base::sync_with_stdio(false);

  int *dummy;
  mgpu::standard_context_t context(false);
  // cuda_bfs(0, 0, dummy, dummy, dummy, dummy, context);

  // run lca
  // int const N = 5;
  // int const root = 0;
  // std::vector<int> parents{-1, 0, 1, 0, 1};
  // int const Q = 5;
  // std::vector<int> queries{0, 2, 1, 4, 3, 2, 1, 2, 0, 4};

  int const N = 20;
  int const root = 11;
  std::vector<int> parents{18, 12, 19, 9, 13, 1, 8, 18, 14, 12, 14, -1, 11, 11, 16, 18, 11, 15, 16, 11};
  int const Q = 10;
  std::vector<int> queries{0, 13, 3, 19, 1, 1, 11, 17, 15, 7, 2, 17, 0, 10, 6, 1, 3, 2, 0, 13};

  mgpu::mem_t<int> cuda_parents = mgpu::to_mem(parents, context);
  mgpu::mem_t<int> cuda_queries = mgpu::to_mem(queries, context);
  mgpu::mem_t<int> cuda_answers = mgpu::fill<int>(0, Q, context);

  // cuda_lca_naive( N, cuda_parents.data(), Q, cuda_queries.data(), cuda_answers.data(), context );
  cuda_lca_inlabel(N, cuda_parents.data(), Q, cuda_queries.data(), cuda_answers.data(), context);

  std::vector<int> correct_answers{11, 11, 1, 11, 18, 11, 16, 11, 11, 11};

  bool isOk = true;
  int *answers = new int[Q];
  mgpu::dtoh(answers, cuda_answers.data(), Q);
  for (int i = 0; i < Q; i++)
  {
    if (answers[i] != correct_answers[i])
      isOk = false;
  }

  if (isOk)
  {
    std::cout << "LCA OK" << std::endl;
  }
  else
  {
    std::cout << "LCA ERROR!" << std::endl;
    for (int i = 0; i < Q; i++)
    {
      std::cout << answers[i] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
