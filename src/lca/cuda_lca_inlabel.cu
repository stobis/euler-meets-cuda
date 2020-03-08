#include <cuda_runtime.h>
#include <iostream>
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/transform.hxx>

#include "commons/commons.h"
#include "commons/cuda_euler_tour.h"
#include "commons/cuda_list_rank.h"
#include "lca/commons.h"
#include "lca/cuda_commons.h"

#define ll long long

using namespace std;
using namespace mgpu;

inline __device__ long long CudaPackEdge(int from, int to);
inline __device__ int CudaUnpackEdgeFrom(long long edge);
inline __device__ int CudaUnpackEdgeTo(long long edge);

inline __device__ int CudaGetEdgeStart(const int *__restrict__ father, int edgeCode);
inline __device__ int CudaGetEdgeEnd(const int *__restrict__ father, int edgeCode);
inline __device__ int CudaGetEdgeCode(int a, bool toFather);
inline __device__ bool isEdgeToFather(int edgeCode);

void cuda_lca_inlabel(int N, const int *parents, int Q, const int *queries,
                      int *answers, mgpu::context_t &context, Logger &logger) {
  Timer timer("Parse Input");
  timer.setPrefix("Converting to crs");

  int root = 0;

  int *devRoot;
  CUCHECK(cudaMalloc((void **)&devRoot, sizeof(int)));
  int *rootArr = new int[1];

  transform(
      [=] MGPU_DEVICE(int thid) {
        if (parents[thid] == -1) {
          devRoot[0] = thid;
        }
      },
      N, context);
  context.synchronize();

  CUCHECK(cudaMemcpy(rootArr, devRoot, sizeof(int), cudaMemcpyDeviceToHost));
  root = rootArr[0];

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

  cuda_euler_tour(N, root, dev_edge_from, dev_edge_to, devEdgeRankFromEulerTour,
                  context);

  int *devEdgeRank;
  CUCHECK(cudaMalloc((void **)&devEdgeRank, sizeof(int) * N * 2));

  transform(
      [=] MGPU_DEVICE(int thid) {
        if (thid == root) {
          devEdgeRank[thid * 2] = -1;
          devEdgeRank[thid * 2 + 1] = N * 2 - 2;
        } else {
          int afterRoot = thid > root;

          devEdgeRank[thid * 2] =
              devEdgeRankFromEulerTour[thid + N - 1 - afterRoot];
          devEdgeRank[thid * 2 + 1] =
              devEdgeRankFromEulerTour[thid - afterRoot];
        }
      },
      N, context);
  CUCHECK(cudaFree(devEdgeRankFromEulerTour));

  timer.measureTime("List Rank");

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
        devAscendant[thid] =
            (1 << l) + devW1Sum[devEdgeRank[CudaGetEdgeCode(thid, 0)]];
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

  timer.setPrefix("Queries");

  // TODO/TOREMOVE
  int batchSize = Q;

  for (int qStart = 0; qStart < Q; qStart += batchSize) {
    int queriesToProcess = min(batchSize, Q - qStart);

    transform(
        [=] MGPU_DEVICE(int thid) {
          int x = queries[thid * 2];
          int y = queries[thid * 2 + 1];

          int inlabelX = devInlabel[x];
          int inlabelY = devInlabel[y];

          if (inlabelX == inlabelY) {
            answers[thid] = devLevel[x] < devLevel[y] ? x : y;
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
            answers[thid] = suspects[0];
          else
            answers[thid] = suspects[1];
        },
        queriesToProcess, context);
  }

  context.synchronize();

  timer.measureTime(Q);
  timer.setPrefix("");
}

#define ll long long
ll CudaPackEdge(int from, int to) { return (((ll)from) << 32) + to; }
int CudaUnpackEdgeFrom(ll edge) { return edge >> 32; }
int CudaUnpackEdgeTo(ll edge) { return edge & ((1 << 32) - 1); }

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
__device__ int CudaGetEdgeCode(int a, bool toFather) {
  return a * 2 + toFather;
}