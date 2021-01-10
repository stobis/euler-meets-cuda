#include <cuda_runtime.h>
#include <iostream>
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/transform.hxx>
#include <string>

#include "euler.h"
#include "lca.h"
#include "timer.hpp"
#include "tree.h"
#include "utils.h"

#define ll long long

using namespace std;
using namespace mgpu;

namespace emc {

inline __device__ long long CudaPackEdge(int from, int to);
inline __device__ int CudaUnpackEdgeFrom(long long edge);
inline __device__ int CudaUnpackEdgeTo(long long edge);

inline __device__ int CudaGetEdgeStart(const int *__restrict__ father, int edgeCode);
inline __device__ int CudaGetEdgeEnd(const int *__restrict__ father, int edgeCode);
inline __device__ int CudaGetEdgeCode(int a, bool toFather);
inline __device__ bool isEdgeToFather(int edgeCode);

void cuda_lca_inlabel(int N, const int *parents, int Q, const int *queries, int *answers, int batchSize,
                      mgpu::context_t &context) {
  Timer timer("CUDA Inlabel");

  int root = 0;

  int *devRoot;
  CUCHECK(cudaMalloc((void **)&devRoot, sizeof(int)));

  transform(
      [=] MGPU_DEVICE(int thid) {
        if (parents[thid] == -1) {
          devRoot[0] = thid;
        }
      },
      N, context);
  context.synchronize();

  CUCHECK(cudaMemcpy(&root, devRoot, sizeof(int), cudaMemcpyDeviceToHost));
  CUCHECK(cudaFree(devRoot));

  int *dev_edge_from;
  int *dev_edge_to;
  CUCHECK(cudaMalloc((void **)&dev_edge_from, sizeof(int) * (N - 1)));
  CUCHECK(cudaMalloc((void **)&dev_edge_to, sizeof(int) * (N - 1)));
  transform(
      [=] MGPU_DEVICE(int thid) {
        if (thid == root)
          return;
        int afterRoot = thid > root;
        dev_edge_from[thid - afterRoot] = thid;
        dev_edge_to[thid - afterRoot] = parents[thid];
      },
      N, context);

  int *devEdgeRankFromEulerTour;
  CUCHECK(cudaMalloc((void **)&devEdgeRankFromEulerTour, sizeof(int) * N * 2));

  cuda_euler_tour(N, root, dev_edge_from, dev_edge_to, devEdgeRankFromEulerTour, context);

  CUCHECK(cudaFree(dev_edge_from));
  CUCHECK(cudaFree(dev_edge_to));

  int *devEdgeRank;
  CUCHECK(cudaMalloc((void **)&devEdgeRank, sizeof(int) * N * 2));

  transform(
      [=] MGPU_DEVICE(int thid) {
        if (thid == root) {
          devEdgeRank[thid * 2] = -1;
          devEdgeRank[thid * 2 + 1] = N * 2 - 2;
        } else {
          int afterRoot = thid > root;

          devEdgeRank[thid * 2] = devEdgeRankFromEulerTour[thid + N - 1 - afterRoot];
          devEdgeRank[thid * 2 + 1] = devEdgeRankFromEulerTour[thid - afterRoot];
        }
      },
      N, context);
  CUCHECK(cudaFree(devEdgeRankFromEulerTour));

  int *devSortedEdges;
  int E = N * 2 - 2;
  CUCHECK(cudaMalloc((void **)&devSortedEdges, sizeof(int) * E));
  transform(
      [=] MGPU_DEVICE(int thid) {
        int edgeRank = devEdgeRank[thid];

        if (edgeRank != -1 && edgeRank != N * 2 - 1)
          devSortedEdges[edgeRank] = thid;
      },
      N * 2, context);

  int *devW1;
  int *devW2;
  int *devW1Sum;
  int *devW2Sum;

  CUCHECK(cudaMalloc((void **)&devW1, sizeof(int) * E));
  CUCHECK(cudaMalloc((void **)&devW1Sum, sizeof(int) * E));
  CUCHECK(cudaMalloc((void **)&devW2, sizeof(int) * E));
  CUCHECK(cudaMalloc((void **)&devW2Sum, sizeof(int) * E));

  transform(
      [=] MGPU_DEVICE(int thid) {
        int edge = devSortedEdges[thid];
        if (isEdgeToFather(edge)) {
          devW1[thid] = 0;
          devW2[thid] = -1;
        } else {
          devW1[thid] = 1;
          devW2[thid] = 1;
        }
      },
      E, context);
  context.synchronize();
  CUCHECK(cudaFree(devSortedEdges));

  scan<scan_type_inc>(devW1, E, devW1Sum, context);
  scan<scan_type_inc>(devW2, E, devW2Sum, context);
  CUCHECK(cudaFree(devW2));

  int *devPreorder;
  int *devPrePlusSize;
  int *devLevel;
  CUCHECK(cudaMalloc((void **)&devPreorder, sizeof(int) * N));
  CUCHECK(cudaMalloc((void **)&devPrePlusSize, sizeof(int) * N));
  CUCHECK(cudaMalloc((void **)&devLevel, sizeof(int) * N));

  transform(
      [=] MGPU_DEVICE(int thid) {
        int codeFromFather = CudaGetEdgeCode(thid, 0);
        int codeToFather = CudaGetEdgeCode(thid, 1);
        if (thid == root) {
          devPreorder[thid] = 1;
          devPrePlusSize[thid] = N;
          devLevel[thid] = 0;
          return;
        }

        int edgeRankFromFather = devEdgeRank[codeFromFather];
        devPreorder[thid] = devW1Sum[edgeRankFromFather] + 1;
        devPrePlusSize[thid] = devW1Sum[devEdgeRank[codeToFather]] + 1;
        devLevel[thid] = devW2Sum[edgeRankFromFather];
      },
      N, context);
  CUCHECK(cudaFree(devW2Sum));

  int *devInlabel;
  CUCHECK(cudaMalloc((void **)&devInlabel, sizeof(int) * N));

  transform(
      [=] MGPU_DEVICE(int thid) {
        int i = 31 - __clz((devPreorder[thid] - 1) ^ (devPrePlusSize[thid]));
        devInlabel[thid] = ((devPrePlusSize[thid]) >> i) << i;
      },
      N, context);

  CUCHECK(cudaFree(devPreorder));
  CUCHECK(cudaFree(devPrePlusSize));

  transform(
      [=] MGPU_DEVICE(int thid) {
        if (thid == root)
          return;
        int f = parents[thid];
        int inLabel = devInlabel[thid];
        int fatherInLabel = devInlabel[f];
        if (inLabel != fatherInLabel) {
          int i = __ffs(inLabel) - 1;
          devW1[devEdgeRank[CudaGetEdgeCode(thid, 0)]] = (1 << i);
          devW1[devEdgeRank[CudaGetEdgeCode(thid, 1)]] = -(1 << i);
        } else {
          devW1[devEdgeRank[CudaGetEdgeCode(thid, 0)]] = 0;
          devW1[devEdgeRank[CudaGetEdgeCode(thid, 1)]] = 0;
        }
      },
      N, context);

  scan<scan_type_inc>(devW1, E, devW1Sum, context);
  CUCHECK(cudaFree(devW1));

  int l = 31 - __builtin_clz(N);

  int *devAscendant;
  CUCHECK(cudaMalloc((void **)&devAscendant, sizeof(int) * N));

  transform(
      [=] MGPU_DEVICE(int thid) {
        if (thid == root) {
          devAscendant[thid] = (1 << l);
          return;
        }
        devAscendant[thid] = (1 << l) + devW1Sum[devEdgeRank[CudaGetEdgeCode(thid, 0)]];
      },
      N, context);
  CUCHECK(cudaFree(devEdgeRank));
  CUCHECK(cudaFree(devW1Sum));

  int *devHead;
  CUCHECK(cudaMalloc((void **)&devHead, sizeof(int) * (N + 1)));

  transform(
      [=] MGPU_DEVICE(int thid) {
        if (thid == root || devInlabel[thid] != devInlabel[parents[thid]]) {
          devHead[devInlabel[thid]] = thid;
        }
      },
      N, context);

  context.synchronize();
  timer.print_and_restart("Preprocessing");

  for (int qStart = 0; qStart < Q; qStart += batchSize) {
    int queriesToProcess = min(batchSize, Q - qStart);

    transform(
        [=] MGPU_DEVICE(int thid) {
          int x = queries[(qStart + thid) * 2];
          int y = queries[(qStart + thid) * 2 + 1];

          int inlabelX = devInlabel[x];
          int inlabelY = devInlabel[y];

          if (inlabelX == inlabelY) {
            answers[qStart + thid] = devLevel[x] < devLevel[y] ? x : y;
            return;
          }
          int i = 31 - __clz(inlabelX ^ inlabelY);

          int common = devAscendant[x] & devAscendant[y];
          common = ((common >> i) << i);

          int j = __ffs(common) - 1;

          int inlabelZ = (inlabelY >> (j)) << (j);
          inlabelZ |= (1 << j);

          int suspects[2];

          for (int a = 0; a < 2; a++) {
            int tmpX;
            if (a == 0)
              tmpX = x;
            else
              tmpX = y;

            if (devInlabel[tmpX] == inlabelZ) {
              suspects[a] = tmpX;
            } else {
              int k = 31 - __clz(devAscendant[tmpX] & ((1 << j) - 1));

              int inlabelW = (devInlabel[tmpX] >> k) << (k);
              inlabelW |= (1 << k);

              int w = devHead[inlabelW];
              suspects[a] = parents[w];
            }
          }

          if (devLevel[suspects[0]] < devLevel[suspects[1]])
            answers[qStart + thid] = suspects[0];
          else
            answers[qStart + thid] = suspects[1];
        },
        queriesToProcess, context);

        context.synchronize();
  }

  CUCHECK(cudaFree(devLevel));
  CUCHECK(cudaFree(devInlabel));
  CUCHECK(cudaFree(devAscendant));
  CUCHECK(cudaFree(devHead));

  timer.print_and_restart("Queries");
  timer.print_overall();
}

// CUDA LCA NAIVE

#define ll long long
#define ull unsigned long long
ll CudaPackEdge(int from, int to) { return (((ll)from) << 32) + to; }
int CudaUnpackEdgeFrom(ll edge) { return edge >> 32; }
int CudaUnpackEdgeTo(ll edge) { return edge & ((1 << 32) - 1); }

void CudaSimpleListRank(int *devRank, int N, int *devNext, context_t &context) {
  int *notAllDone;
  cudaMallocHost(&notAllDone, sizeof(int));

  ull *devRankNext;
  int *devNotAllDone;

  CUCHECK(cudaMalloc((void **)&devRankNext, sizeof(ull) * N));
  CUCHECK(cudaMalloc((void **)&devNotAllDone, sizeof(int)));

  transform([] MGPU_DEVICE(int thid, ull *devRankNext,
                           const int *devNext) { devRankNext[thid] = (((ull)0) << 32) + devNext[thid] + 1; },
            N, context, devRankNext, devNext);

  const int loopsWithoutSync = 5;

  do {
    transform(
        [] MGPU_DEVICE(int thid, int loopsWithoutSync, ull *devRankNext, int *devNotAllDone) {
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

    CUCHECK(cudaMemcpy(notAllDone, devNotAllDone, sizeof(int), cudaMemcpyDeviceToHost));
  } while (*notAllDone);

  transform([] MGPU_DEVICE(int thid, const ull *devRankNext,
                           int *devRank) { devRank[thid] = devRankNext[thid] >> 32; },
            N, context, devRankNext, devRank);

  context.synchronize();

  CUCHECK(cudaFree(devRankNext));
  CUCHECK(cudaFree(devNotAllDone));
}

__device__ int CudaGetEdgeStart(const int *__restrict__ father, int edgeCode) {
  if (edgeCode % 2)
    return edgeCode / 2;
  else
    return father[edgeCode / 2];
}
__device__ int CudaGetEdgeEnd(const int *__restrict__ father, int edgeCode) {
  return CudaGetEdgeStart(father, edgeCode ^ 1);
}
__device__ bool isEdgeToFather(int edgeCode) { return edgeCode % 2; }
__device__ int CudaGetEdgeCode(int a, bool toFather) { return a * 2 + toFather; }

void cuda_lca_naive(int N, const int *parents, int Q, const int *queries, int *answers, int batchSize,
                    mgpu::context_t &context) {
  Timer timer("CUDA Naive");

  int *devDepth;
  int *devNext;

  CUCHECK(cudaMalloc((void **)&devDepth, sizeof(int) * N));
  CUCHECK(cudaMalloc((void **)&devNext, sizeof(int) * N));

  transform(
      [=] MGPU_DEVICE(int thid) {
        devNext[thid] = parents[thid];
        devDepth[thid] = 0;
      },
      N, context);
  context.synchronize();

  CudaSimpleListRank(devDepth, N, devNext, context);
  context.synchronize();

  timer.print_and_restart("Preprocessing");

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
  context.synchronize();

  CUCHECK(cudaFree(devDepth));
  CUCHECK(cudaFree(devNext));

  timer.print_and_restart("Queries");
  timer.print_overall();
}

} // namespace emc
