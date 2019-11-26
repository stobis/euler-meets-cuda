#include "bridges/cuda_algorithms.h"
#include "lca/cuda_algorithms.h"

#include "lca_commons.h"
#include "lca_cuda_commons.h"

#include<moderngpu/context.hxx>

int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false);

    int * dummy;
    mgpu::standard_context_t context(false);
    //cuda_bfs(0, 0, dummy, dummy, dummy, dummy, context);

     TestCase tc;
     if ( argc == 1 )
     {
       tc = readFromStdIn();
     }
     else
     {
       tc = readFromFile( argv[1] );
     }


  const int V = tc.tree.V;
  int Q = tc.q.N;
  
  int *devFather;
  int *devQueries;
  int *devAnswers;

  CUCHECK(cudaMalloc((void **) &devFather, sizeof(int)*V));
  CUCHECK( cudaMemcpy( devFather, tc.tree.father.data(), 
          sizeof( int ) * V, cudaMemcpyHostToDevice ) );


  CUCHECK(cudaMalloc((void**) &devQueries, sizeof( int ) * Q * 2 ) );
  CUCHECK( cudaMemcpy( devQueries, tc.q.tab.data(), 
          sizeof( int ) * Q * 2, cudaMemcpyHostToDevice ) );
  CUCHECK(cudaMalloc((void**) &devAnswers, sizeof( int ) * Q ) );

  cuda_lca_naive(tc.tree.V , devFather, tc.q.N, devQueries,
          devAnswers, context);

  int *answers = new int[Q];
  CUCHECK( cudaMemcpy( answers, devAnswers, 
          sizeof( int ) * Q, cudaMemcpyDeviceToHost ) );

   
     if ( argc < 3 )
     {
       writeAnswersToStdOut( Q, answers );
     }
     else
     {
       writeAnswersToFile( Q, answers, argv[2] );
     }

  return 0;
}
