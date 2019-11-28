#include <iostream>

#include "bridges/cuda_algorithms.h"
#include "lca/cuda_algorithms.h"

#include <moderngpu/context.hxx>
#include <moderngpu/memory.hxx>

int main( int argc, char *argv[] )
{
  std::ios_base::sync_with_stdio( false );

  int *dummy;
  mgpu::standard_context_t context( false );
  // cuda_bfs(0, 0, dummy, dummy, dummy, dummy, context);


  // run lca
  int const N = 5;
  int const root = 2;
  std::vector<int> parents{1, 2, -1, 1, 0};
  int const Q = 5;
  std::vector<int> queries{4, 3, 0, 4, 3, 0, 4, 2, 2, 2};

  mgpu::mem_t<int> cuda_parents = mgpu::to_mem( parents, context );
  mgpu::mem_t<int> cuda_queries = mgpu::to_mem( queries, context );
  mgpu::mem_t<int> cuda_answers = mgpu::fill<int>( 0, Q, context );

  // cuda_lca_naive( N, cuda_parents.data(), Q, cuda_queries.data(), cuda_answers.data(), context );
  cuda_lca_inlabel( N, cuda_parents.data(), Q, cuda_queries.data(), cuda_answers.data(), context );

  int *answers = new int[Q];
  mgpu::dtoh( answers, cuda_answers.data(), Q );
  for ( int i = 0; i < Q; i++ )
  {
    std::cout << answers[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}
