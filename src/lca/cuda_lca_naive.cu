#include "lca/commons.h"
#include "lca/cuda_commons.h"
#include "lca/cuda_lca_naive.h"

#include <cuda_runtime.h>
#include <iostream>
#include <moderngpu/context.hxx>
#include <moderngpu/transform.hxx>

using namespace std;
using namespace mgpu;

__global__ void cuCalcQueries(int Q, const int *__restrict__ parents,
                              const int *__restrict__ devDepth,
                              const int *__restrict__ queries, int *answers);

void cuda_lca_naive(int N, const int *parents, int Q, const int *queries,
                    int *answers, mgpu::context_t &context) {

  Timer timer = Timer("Parse Input");

  int *devDepth;
  int *devNext;

  CUCHECK(cudaMalloc((void **)&devDepth, sizeof(int) * N));
  CUCHECK(cudaMalloc((void **)&devNext, sizeof(int) * N));

  timer.measureTime("Cuda Allocs");

  int threadsPerBlockX = 1024;
  int blockPerGridX = (N + threadsPerBlockX - 1) / threadsPerBlockX;

  transform(
      [=] MGPU_DEVICE(int thid) {
        devNext[thid] = parents[thid];
        devDepth[thid] = 0;
      },
      N, context);
  context.synchronize();

  timer.measureTime("Copy Input and Init data");

  CudaSimpleListRank(devDepth, N, devNext, context);
  context.synchronize();

  timer.setPrefix("Queries");

  blockPerGridX = (Q + threadsPerBlockX - 1) / threadsPerBlockX;

  transform(
      [=] MGPU_DEVICE(int thid) {
        int p = queries[thid * 2];
        int q = queries[thid * 2 + 1];

        if (p == q)
          answers[thid] = p;

        while (devDepth[p] != devDepth[q]) {
          if (devDepth[p] > devDepth[q])
            p = parents[p];
          else
            q = parents[q];
        }

        while (p != q) {
          p = parents[p];
          q = parents[q];
        }

        answers[thid] = p;
      },
      Q, context);
  CUCHECK(cudaDeviceSynchronize());

  CUCHECK(cudaFree(devDepth));
  CUCHECK(cudaFree(devNext));

  context.synchronize();
  timer.measureTime(Q);
}