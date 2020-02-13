#include "lca/cuda_lca_naive.h"
#include "lca/commons.h"
#include "lca/cuda_commons.h"

#include <cuda_runtime.h>
#include <iostream>
#include <moderngpu/context.hxx>
#include <moderngpu/transform.hxx>

using namespace std;
using namespace mgpu;

__global__ void cuCalcQueries( int Q,
                               const int *__restrict__ father,
                               const int *__restrict__ depth,
                               const int *__restrict__ queries,
                               int *answers );

void cuda_lca_naive( int N, const int *parents, int Q, 
        const int *queries, int *answers, 
        mgpu::context_t &context )
{

  Timer timer = Timer( "Parse Input" );

  //   cudaSetDevice( 0 );
    //   timer.setPrefix( "Preprocessing" );

  //   int *devFather;
  int *devDepth;
  int *devNext;
  const int *devQueries;
  int *devAnswers;

  //   const int V = tc.tree.V;
  const int V = N;

  //   CUCHECK( cudaMalloc( (void **) &devFather, sizeof( int ) * V ) );
  CUCHECK( cudaMalloc( (void **) &devDepth, sizeof( int ) * V ) );
  CUCHECK( cudaMalloc( (void **) &devNext, sizeof( int ) * V ) );

  timer.measureTime( "Cuda Allocs" );

  //   CUCHECK( cudaMemcpy( devFather, tc.tree.father.data(), sizeof( int ) * V, cudaMemcpyHostToDevice ) );
  const int *devFather = parents;

  int threadsPerBlockX = 1024;
  int blockPerGridX = ( V + threadsPerBlockX - 1 ) / threadsPerBlockX;

  transform(
      [] MGPU_DEVICE( int thid, const int *devFather, int *devDepth, int *devNext ) {
        devNext[thid] = devFather[thid];
        devDepth[thid] = 0;
      },
      V,
      context,
      devFather,
      devDepth,
      devNext );

  context.synchronize();

  timer.measureTime( "Copy Input and Init data" );

  CudaSimpleListRank( devDepth, V, devNext, context );
  context.synchronize();

  timer.setPrefix( "Queries" );

  devQueries = queries;
  devAnswers = answers;


  timer.measureTime( "Copy Queries to Dev" );

  blockPerGridX = ( Q + threadsPerBlockX - 1 ) / threadsPerBlockX;

  cuCalcQueries<<<blockPerGridX, threadsPerBlockX>>>( Q, devFather, devDepth, devQueries, devAnswers );
  CUCHECK( cudaDeviceSynchronize() );

  timer.measureTime( Q );

  context.synchronize();
  timer.measureTime( "Copy answers to Host" );
  timer.setPrefix( "Write Output" );

  //   if ( argc < 3 )
  //   {
  //     writeAnswersToStdOut( Q, answers );
  //   }
  //   else
  //   {
  //     writeAnswersToFile( Q, answers, argv[2] );
  //   }

  timer.setPrefix( "" );

  return;
}

__global__ void cuCalcQueries( int Q,
                               const int *__restrict__ father,
                               const int *__restrict__ depth,
                               const int *__restrict__ queries,
                               int *answers )
{
  int thid = ( blockIdx.x * blockDim.x ) + threadIdx.x;

  if ( thid >= Q ) return;

  int p = queries[thid * 2];
  int q = queries[thid * 2 + 1];

  if ( p == q ) answers[thid] = p;

  while ( depth[p] != depth[q] )
  {
    if ( depth[p] > depth[q] )
      p = father[p];
    else
      q = father[q];
  }

  while ( p != q )
  {
    p = father[p];
    q = father[q];
  }

  answers[thid] = p;
}
