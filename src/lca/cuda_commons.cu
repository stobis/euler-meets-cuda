#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <moderngpu/transform.hxx>

#include "lca/commons.h"
#include "lca/cuda_commons.h"

#define ull unsigned long long

using namespace std;
using namespace mgpu;

void CudaSimpleListRank(int *devRank, int N, int *devNext, context_t &context) {
  int *notAllDone;
  cudaMallocHost(&notAllDone, sizeof(int));

  ull *devRankNext;
  int *devNotAllDone;

  CUCHECK(cudaMalloc((void **)&devRankNext, sizeof(ull) * N));
  CUCHECK(cudaMalloc((void **)&devNotAllDone, sizeof(int)));

  transform(
      [] MGPU_DEVICE(int thid, ull *devRankNext, const int *devNext) {
        devRankNext[thid] = (((ull)0) << 32) + devNext[thid] + 1;
      },
      N, context, devRankNext, devNext);

  const int loopsWithoutSync = 5;

  do {
    transform(
        [] MGPU_DEVICE(int thid, int loopsWithoutSync, ull *devRankNext,
                       int *devNotAllDone) {
          ull rankNext = devRankNext[thid];
          for (int i = 0; i < loopsWithoutSync; i++) {
            if (thid == 0)
              *devNotAllDone = 0;

            int rank = rankNext >> 32;
            int nxt = rankNext - 1;

            if (nxt != -1) {
              ull grandNext = devRankNext[nxt];

              rank += (grandNext >> 32) + 1;
              nxt = grandNext - 1;

              rankNext = (((ull)rank) << 32) + nxt + 1;
              atomicExch(devRankNext + thid, rankNext);

              if (i == loopsWithoutSync - 1)
                *devNotAllDone = 1;
            }
          }
        },
        N, context, loopsWithoutSync, devRankNext, devNotAllDone);

    context.synchronize();

    CUCHECK(cudaMemcpy(notAllDone, devNotAllDone, sizeof(int),
                       cudaMemcpyDeviceToHost));
  } while (*notAllDone);

  transform(
      [] MGPU_DEVICE(int thid, const ull *devRankNext, int *devRank) {
        devRank[thid] = devRankNext[thid] >> 32;
      },
      N, context, devRankNext, devRank);

  cudaFree(notAllDone);
  cudaFree(devRankNext);
  CUCHECK(cudaFree(devNotAllDone));
}

void CudaAssert(cudaError_t error, const char *code, const char *file,
                int line) {
  if (error != cudaSuccess) {
    cerr << "Cuda error :" << code << ", " << file << ":" << error << endl;
    exit(1);
  }
}

void CudaPrintTab(const int *tab, int size) {
  int *tmp = (int *)malloc(sizeof(int) * size);
  CUCHECK(cudaMemcpy(tmp, tab, sizeof(int) * size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < size; i++) {
    cerr << (tmp[i] < 10 && tmp[i] >= 0 ? " " : "") << tmp[i] << " ";
  }
  cerr << endl;

  free(tmp);
}