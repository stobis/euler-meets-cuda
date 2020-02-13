#include <cuda_runtime.h>
#include <iostream>
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/transform.hxx>

#include "lca/commons.h"
#include "lca/cuda_commons.h"
#include "commons/cuda_euler_tour.h"
#include "commons/cuda_list_rank.h"
#include "commons/commons.h"

#define ll long long

using namespace std;
using namespace mgpu;

__device__ long long CudaPackEdge(int from, int to);
__device__ int CudaUnpackEdgeFrom(long long edge);
__device__ int CudaUnpackEdgeTo(long long edge);

__device__ int CudaGetEdgeStart(const int *__restrict__ father, int edgeCode);
__device__ int CudaGetEdgeEnd(const int *__restrict__ father, int edgeCode);
__device__ int CudaGetEdgeCode(int a, bool toFather);
__device__ bool isEdgeToFather(int edgeCode);

const int measureTimeDebug = false;

void cuda_lca_inlabel(int N, const int *parents, int Q, const int *queries,
                      int *answers, mgpu::context_t &context, Logger &logger)
{
  Timer timer("Parse Input");

  timer.setPrefix("Converting to crs");

  const int V = N;
  int root = 0;

  int *devRoot;
  CUCHECK(cudaMalloc((void **)&devRoot, sizeof(int)));
  int *rootArr = new int[1];

  transform(
      [] MGPU_DEVICE(int thid, const int *devParent, int *devRoot) {
        if (devParent[thid] == -1)
        {
          devRoot[0] = thid;
        }
      },
      V, context, parents, devRoot);

  context.synchronize();

  CUCHECK(cudaMemcpy(rootArr, devRoot, sizeof(int), cudaMemcpyDeviceToHost));
  root = rootArr[0];

  int *dev_edge_from;
  int *dev_edge_to;
  CUCHECK(cudaMalloc((void **)&dev_edge_from, sizeof(int) * (V - 1)));
  CUCHECK(cudaMalloc((void **)&dev_edge_to, sizeof(int) * (V - 1)));
  transform([=] MGPU_DEVICE(int thid) {
    if (thid == root)
      return;
    int afterRoot = thid > root;
    dev_edge_from[thid - afterRoot] = thid;
    dev_edge_to[thid - afterRoot] = parents[thid];
  },
            V, context);

  int *devEdgeRankFromEulerTour;
  CUCHECK(cudaMalloc((void **)&devEdgeRankFromEulerTour, sizeof(int) * V * 2));

  cuda_euler_tour(V, root, dev_edge_from, dev_edge_to, devEdgeRankFromEulerTour, context);

  int *devEdgeRank;
  CUCHECK(cudaMalloc((void **)&devEdgeRank, sizeof(int) * V * 2));

  transform([=] MGPU_DEVICE(int thid) {
    if (thid == root)
    {
      devEdgeRank[thid * 2] = 0;
      devEdgeRank[thid * 2 + 1] = V * 2 - 1;
    }
    else
    {
      int afterRoot = thid > root;

      devEdgeRank[thid * 2] = devEdgeRankFromEulerTour[thid + V - 1 - afterRoot] + 1;
      devEdgeRank[thid * 2 + 1] = devEdgeRankFromEulerTour[thid - afterRoot] + 1;
    }
  },
            V, context);

  const int *devFather = parents;

  timer.measureTime("List Rank");

  int *devSortedEdges;
  int E = V * 2 - 2;

  CUCHECK(cudaMalloc((void **)&devSortedEdges, sizeof(int) * E));

  transform(
      [] MGPU_DEVICE(int thid, int V, int *devEdgeRank, int *devSortedEdges) {
        int edgeRank = devEdgeRank[thid] - 1;

        devEdgeRank[thid] = edgeRank;

        if (edgeRank == -1 || edgeRank == V * 2 - 1)
          return; // edges from root

        devSortedEdges[edgeRank] = thid;
      },
      V * 2, context, V, devEdgeRank, devSortedEdges);

  int *devW1;
  int *devW2;
  int *devW1Sum;
  int *devW2Sum;

  CUCHECK(cudaMalloc((void **)&devW1, sizeof(int) * E));

  cerr << sizeof(int) * E << endl;

  CUCHECK(cudaMalloc((void **)&devW1Sum, sizeof(int) * E));
  CUCHECK(cudaMalloc((void **)&devW2, sizeof(int) * E));
  CUCHECK(cudaMalloc((void **)&devW2Sum, sizeof(int) * E));

  if (measureTimeDebug)
  {
    context.synchronize();
    timer.measureTime("Inlabel allocs");
  }

  transform(
      [] MGPU_DEVICE(int thid, int *devW1, int *devW2,
                     const int *devSortedEdges) {
        int edge = devSortedEdges[thid];
        if (isEdgeToFather(edge))
        {
          devW1[thid] = 0;
          devW2[thid] = -1;
        }
        else
        {
          devW1[thid] = 1;
          devW2[thid] = 1;
        }
      },
      E, context, devW1, devW2, devSortedEdges);

  context.synchronize();
  CUCHECK(cudaFree(devSortedEdges));

  scan<scan_type_inc>(devW1, E, devW1Sum, context);
  scan<scan_type_inc>(devW2, E, devW2Sum, context);
  CUCHECK(cudaFree(devW2));

  if (measureTimeDebug)
  {
    context.synchronize();
    timer.measureTime("W1 W2 scans");
  }

  int *devPreorder;
  int *devPrePlusSize;
  int *devLevel;

  CUCHECK(cudaMalloc((void **)&devPreorder, sizeof(int) * V));
  CUCHECK(cudaMalloc((void **)&devPrePlusSize, sizeof(int) * V));
  CUCHECK(cudaMalloc((void **)&devLevel, sizeof(int) * V));

  transform(
      [] MGPU_DEVICE(int thid, int V, int root, int *devPreorder,
                     int *devPrePlusSize, int *devLevel, const int *devEdgeRank,
                     const int *devW1Sum, const int *devW2Sum) {
        int codeFromFather = CudaGetEdgeCode(thid, 0);
        int codeToFather = CudaGetEdgeCode(thid, 1);
        if (thid == root)
        {
          devPreorder[thid] = 1;
          devPrePlusSize[thid] = V;
          devLevel[thid] = 0;
          return;
        }

        int edgeRankFromFather = devEdgeRank[codeFromFather];
        devPreorder[thid] = devW1Sum[edgeRankFromFather] + 1;
        devPrePlusSize[thid] = devW1Sum[devEdgeRank[codeToFather]] + 1;
        devLevel[thid] = devW2Sum[edgeRankFromFather];
      },
      V, context, V, root, devPreorder, devPrePlusSize, devLevel, devEdgeRank,
      devW1Sum, devW2Sum);

  CUCHECK(cudaFree(devW2Sum));

  if (measureTimeDebug)
  {
    context.synchronize();
    timer.measureTime("Pre PrePlusSize, Level");
  }

  int *devInlabel;

  CUCHECK(cudaMalloc((void **)&devInlabel, sizeof(int) * V));

  transform(
      [] MGPU_DEVICE(int thid, int *devInlabel, const int *devPreorder,
                     const int *devPrePlusSize) {
        int i = 31 - __clz((devPreorder[thid] - 1) ^ (devPrePlusSize[thid]));
        devInlabel[thid] = ((devPrePlusSize[thid]) >> i) << i;
      },
      V, context, devInlabel, devPreorder, devPrePlusSize);

  CUCHECK(cudaFree(devPreorder));
  CUCHECK(cudaFree(devPrePlusSize));

  transform(
      [] MGPU_DEVICE(int thid, int root, const int *devFather,
                     const int *devInlabel, const int *devEdgeRank,
                     int *devW1) {
        if (thid == root)
          return;
        int f = devFather[thid];
        int inLabel = devInlabel[thid];
        int fatherInLabel = devInlabel[f];
        if (inLabel != fatherInLabel)
        {
          int i = __ffs(inLabel) - 1;
          devW1[devEdgeRank[CudaGetEdgeCode(thid, 0)]] = (1 << i);
          devW1[devEdgeRank[CudaGetEdgeCode(thid, 1)]] = -(1 << i);
        }
        else
        {
          devW1[devEdgeRank[CudaGetEdgeCode(thid, 0)]] = 0;
          devW1[devEdgeRank[CudaGetEdgeCode(thid, 1)]] = 0;
        }
      },
      V, context, root, devFather, devInlabel, devEdgeRank, devW1);

  scan<scan_type_inc>(devW1, E, devW1Sum, context);
  CUCHECK(cudaFree(devW1));

  int l = 31 - __builtin_clz(V);

  int *devAscendant;
  CUCHECK(cudaMalloc((void **)&devAscendant, sizeof(int) * V));

  transform(
      [] MGPU_DEVICE(int thid, int root, int l, int *devAscendant,
                     const int *devW1Sum, const int *devEdgeRank) {
        if (thid == root)
        {
          devAscendant[thid] = (1 << l);
          return;
        }
        devAscendant[thid] =
            (1 << l) + devW1Sum[devEdgeRank[CudaGetEdgeCode(thid, 0)]];
      },
      V, context, root, l, devAscendant, devW1Sum, devEdgeRank);

  CUCHECK(cudaFree(devEdgeRank));
  CUCHECK(cudaFree(devW1Sum));

  if (measureTimeDebug)
  {
    context.synchronize();
    timer.measureTime("Ascendant scan and calculation");
  }

  int *devHead;
  CUCHECK(cudaMalloc((void **)&devHead, sizeof(int) * (V + 1)));

  transform(
      [] MGPU_DEVICE(int thid, int root, const int *devInlabel,
                     const int *devFather, int *devHead) {
        if (thid == root || devInlabel[thid] != devInlabel[devFather[thid]])
        {
          devHead[devInlabel[thid]] = thid;
        }
      },
      V, context, root, devInlabel, devFather, devHead);

  context.synchronize();

  if (measureTimeDebug)
    timer.measureTime("Head");

  timer.setPrefix("Queries");

  int batchSize = Q;

  const int *devQueries = queries;

  int *devAnswers = answers;

  for (int qStart = 0; qStart < Q; qStart += batchSize)
  {
    int queriesToProcess = min(batchSize, Q - qStart);

    transform(
        [] MGPU_DEVICE(int thid, const int *devQueries, const int *devInlabel,
                       const int *devLevel, const int *devAscendant,
                       const int *devFather, const int *devHead,
                       int *devAnswers) {
          int x = devQueries[thid * 2];
          int y = devQueries[thid * 2 + 1];

          int inlabelX = devInlabel[x];
          int inlabelY = devInlabel[y];

          if (inlabelX == inlabelY)
          {
            devAnswers[thid] = devLevel[x] < devLevel[y] ? x : y;
            return;
          }
          int i = 31 - __clz(inlabelX ^ inlabelY);

          int common = devAscendant[x] & devAscendant[y];
          common = ((common >> i) << i);

          int j = __ffs(common) - 1;

          int inlabelZ = (inlabelY >> (j)) << (j);
          inlabelZ |= (1 << j);

          int suspects[2];

          for (int a = 0; a < 2; a++)
          {
            int tmpX;
            if (a == 0)
              tmpX = x;
            else
              tmpX = y;

            if (devInlabel[tmpX] == inlabelZ)
            {
              suspects[a] = tmpX;
            }
            else
            {
              int k = 31 - __clz(devAscendant[tmpX] & ((1 << j) - 1));

              int inlabelW = (devInlabel[tmpX] >> k) << (k);
              inlabelW |= (1 << k);

              int w = devHead[inlabelW];
              suspects[a] = devFather[w];
            }
          }

          if (devLevel[suspects[0]] < devLevel[suspects[1]])
            devAnswers[thid] = suspects[0];
          else
            devAnswers[thid] = suspects[1];
        },
        queriesToProcess, context, devQueries, devInlabel, devLevel,
        devAscendant, devFather, devHead, devAnswers);
  }

  context.synchronize();

  timer.measureTime(Q);
  timer.setPrefix("Write Output");
  timer.setPrefix("");
}

#define ll long long
ll CudaPackEdge(int from, int to)
{
  return (((ll)from) << 32) + to;
}
int CudaUnpackEdgeFrom(ll edge) { return edge >> 32; }
int CudaUnpackEdgeTo(ll edge) { return edge & ((1 << 32) - 1); }

__device__ int CudaGetEdgeStart(const int *__restrict__ father, int edgeCode)
{
  if (edgeCode % 2)
    return edgeCode / 2;
  else
    return father[edgeCode / 2];
}
__device__ int CudaGetEdgeEnd(const int *__restrict__ father, int edgeCode)
{
  return CudaGetEdgeStart(father, edgeCode ^ 1);
}
__device__ bool isEdgeToFather(int edgeCode) { return edgeCode % 2; }
__device__ int CudaGetEdgeCode(int a, bool toFather)
{
  return a * 2 + toFather;
}