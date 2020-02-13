#include <curand_kernel.h>
#include <iostream>
#include <moderngpu/transform.hxx>

#include "lca/cuda_commons.h"
#include "commons/cuda_list_rank.h"

using namespace std;
using namespace mgpu;

__device__ int cuAbs( int i )
{
  return i < 0 ? -i : i;
}

void cuda_list_rank( int N, int head, const int *devNextSrc, int *devRank, context_t &context )
{
  int s;
  if ( N >= 100000 )
  {
    s = sqrt( N ) * 1.6;  // Based on experimental results for GTX 980.
  }
  else
    s = N / 100;
  if ( s == 0 ) s = 1;

  int *devNext;
  CUCHECK( cudaMalloc( (void **) &devNext, sizeof( int ) * ( N ) ) );
  transform(
      [=] MGPU_DEVICE( int i, const int *devNextSrc, int *devNext ) {
        devNext[i] = devNextSrc[i];
        if ( devNextSrc[i] == head ) devNext[i] = -1;
      },
      N,
      context,
      devNextSrc,
      devNext );

  transform( [=] MGPU_DEVICE( int i ) { devRank[i] = 0; }, N, context );


  int *devSum;
  CUCHECK( cudaMalloc( (void **) &devSum, sizeof( int ) * ( s + 1 ) ) );
  int *devSublistHead;
  CUCHECK( cudaMalloc( (void **) &devSublistHead, sizeof( int ) * ( s + 1 ) ) );
  int *devSublistId;
  CUCHECK( cudaMalloc( (void **) &devSublistId, sizeof( int ) * N ) );
  int *devLast;
  CUCHECK( cudaMalloc( (void **) &devLast, sizeof( int ) * ( s + 1 ) ) );

  transform(
      [] MGPU_DEVICE( int i, int N, int s, int head, int *devNext, int *devSublistHead ) {
        curandState state;
        curand_init( 123, i, 0, &state );

        int p = i * ( N / s );
        int q = min( p + N / s, N );

        int splitter;
        do
        {
          splitter = ( cuAbs( curand( &state ) ) % ( q - p ) ) + p;
        } while ( devNext[splitter] == -1 );

        devSublistHead[i + 1] = devNext[splitter];
        devNext[splitter] = -i - 2;  // To avoid confusion with -1

        if ( i == 0 )
        {
          devSublistHead[0] = head;
        }
      },
      s,
      context,
      N,
      s,
      head,
      devNext,
      devSublistHead );

  transform(
      [] MGPU_DEVICE( int thid,
                      const int *devSublistHead,
                      const int *devNext,
                      int *devRank,
                      int *devSum,
                      int *devLast,
                      int *devSublistId ) {
        int current = devSublistHead[thid];
        int counter = 0;

        while ( current >= 0 )
        {
          devRank[current] = counter++;

          int n = devNext[current];

          if ( n < 0 )
          {
            devSum[thid] = counter;
            devLast[thid] = current;
          }

          devSublistId[current] = thid;
          current = n;
        }
      },
      s + 1,
      context,
      devSublistHead,
      devNext,
      devRank,
      devSum,
      devLast,
      devSublistId );

  transform(
      [] MGPU_DEVICE( int thid, int head, int s, const int *devNext, const int *devLast, int *devSum ) {
        int tmpSum = 0;
        int current = head;
        int currentSublist = 0;
        for ( int i = 0; i <= s; i++ )
        {
          tmpSum += devSum[currentSublist];
          devSum[currentSublist] = tmpSum - devSum[currentSublist];

          current = devLast[currentSublist];
          currentSublist = -devNext[current] - 1;
        }
      },
      1,
      context,
      head,
      s,
      devNext,
      devLast,
      devSum );

  transform(
      [] MGPU_DEVICE( int thid, const int *devSublistId, const int *devSum, int *devRank ) {
        int sublistId = devSublistId[thid];
        devRank[thid] += devSum[sublistId];
      },
      N,
      context,
      devSublistId,
      devSum,
      devRank );

  CUCHECK( cudaFree( devNext ) );
  CUCHECK( cudaFree( devSum ) );
  CUCHECK( cudaFree( devSublistHead ) );
  CUCHECK( cudaFree( devSublistId ) );
  CUCHECK( cudaFree( devLast ) );

  context.synchronize();
}