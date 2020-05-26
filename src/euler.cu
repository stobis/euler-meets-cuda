#include <iostream>

#include <curand_kernel.h>
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/transform.hxx>

#include "euler.h"
#include "utils.h"

using namespace std;
using namespace mgpu;

__device__ int cuAbs(int i) { return i < 0 ? -i : i; }

void cuda_list_rank(int N, int head, const int *devNextSrc, int *devRank, context_t &context) {
  int s;
  if (N >= 100000) {
    s = sqrt(N) * 1.6; // Based on experimental results for GTX 980.
  } else
    s = N / 100;
  if (s == 0)
    s = 1;

  int *devNext;
  CUCHECK(cudaMalloc((void **)&devNext, sizeof(int) * (N)));
  transform(
      [=] MGPU_DEVICE(int i, const int *devNextSrc, int *devNext) {
        devNext[i] = devNextSrc[i];
        if (devNextSrc[i] == head)
          devNext[i] = -1;
      },
      N, context, devNextSrc, devNext);

  transform([=] MGPU_DEVICE(int i) { devRank[i] = 0; }, N, context);

  int *devSum;
  CUCHECK(cudaMalloc((void **)&devSum, sizeof(int) * (s + 1)));
  int *devSublistHead;
  CUCHECK(cudaMalloc((void **)&devSublistHead, sizeof(int) * (s + 1)));
  int *devSublistId;
  CUCHECK(cudaMalloc((void **)&devSublistId, sizeof(int) * N));
  int *devLast;
  CUCHECK(cudaMalloc((void **)&devLast, sizeof(int) * (s + 1)));

  transform(
      [] MGPU_DEVICE(int i, int N, int s, int head, int *devNext, int *devSublistHead) {
        curandState state;
        curand_init(123, i, 0, &state);

        int p = i * (N / s);
        int q = min(p + N / s, N);

        int splitter;
        do {
          splitter = (cuAbs(curand(&state)) % (q - p)) + p;
        } while (devNext[splitter] == -1);

        devSublistHead[i + 1] = devNext[splitter];
        devNext[splitter] = -i - 2; // To avoid confusion with -1

        if (i == 0) {
          devSublistHead[0] = head;
        }
      },
      s, context, N, s, head, devNext, devSublistHead);

  transform(
      [] MGPU_DEVICE(int thid, const int *devSublistHead, const int *devNext, int *devRank, int *devSum,
                     int *devLast, int *devSublistId) {
        int current = devSublistHead[thid];
        int counter = 0;

        while (current >= 0) {
          devRank[current] = counter++;

          int n = devNext[current];

          if (n < 0) {
            devSum[thid] = counter;
            devLast[thid] = current;
          }

          devSublistId[current] = thid;
          current = n;
        }
      },
      s + 1, context, devSublistHead, devNext, devRank, devSum, devLast, devSublistId);

  transform(
      [] MGPU_DEVICE(int thid, int head, int s, const int *devNext, const int *devLast, int *devSum) {
        int tmpSum = 0;
        int current = head;
        int currentSublist = 0;
        for (int i = 0; i <= s; i++) {
          tmpSum += devSum[currentSublist];
          devSum[currentSublist] = tmpSum - devSum[currentSublist];

          current = devLast[currentSublist];
          currentSublist = -devNext[current] - 1;
        }
      },
      1, context, head, s, devNext, devLast, devSum);

  transform(
      [] MGPU_DEVICE(int thid, const int *devSublistId, const int *devSum, int *devRank) {
        int sublistId = devSublistId[thid];
        devRank[thid] += devSum[sublistId];
      },
      N, context, devSublistId, devSum, devRank);

  CUCHECK(cudaFree(devNext));
  CUCHECK(cudaFree(devSum));
  CUCHECK(cudaFree(devSublistHead));
  CUCHECK(cudaFree(devSublistId));
  CUCHECK(cudaFree(devLast));

  context.synchronize();
}

void cuda_euler_tour(int N, int root, const int *edges_from_input, const int *edges_to_input,
                     int *rank_to_output, mgpu::context_t &context) {
  int E = N * 2 - 2;

  int *edges_to;
  int *edges_from;
  CUCHECK(cudaMalloc((void **)&edges_to, sizeof(int) * E));
  CUCHECK(cudaMalloc((void **)&edges_from, sizeof(int) * E));
  int *index;
  CUCHECK(cudaMalloc((void **)&index, sizeof(int) * E));

  transform(
      [=] MGPU_DEVICE(int thid) {
        edges_from[thid + N - 1] = edges_to[thid] = edges_to_input[thid];
        edges_to[thid + N - 1] = edges_from[thid] = edges_from_input[thid];
      },
      N - 1, context);

  transform([=] MGPU_DEVICE(int thid) { index[thid] = thid; }, E, context);

  mergesort(
      index, E,
      [=] MGPU_DEVICE(int a, int b) {
        int fa = edges_from[a];
        int ta = edges_to[a];
        int fb = edges_from[b];
        int tb = edges_to[b];
        if (fa != fb)
          return fa < fb;
        return ta < tb;
      },
      context);

  int *next;
  CUCHECK(cudaMalloc((void **)&next, sizeof(int) * E));
  transform([=] MGPU_DEVICE(int thid) { next[thid] = -1; }, E, context);

  int *first;
  CUCHECK(cudaMalloc((void **)&first, sizeof(int) * N));

  transform(
      [=] MGPU_DEVICE(int thid) {
        int f = edges_from[index[thid]];
        int t = edges_to[index[thid]];

        if (thid == 0) {
          first[f] = index[thid];
          return;
        }

        int pf = edges_from[index[thid - 1]];
        int pt = edges_to[index[thid - 1]];

        if (f != pf) {
          first[f] = index[thid];
        } else {
          next[index[thid - 1]] = index[thid];
        }
      },
      E, context);
  context.synchronize();

  CUCHECK(cudaFree(edges_to));
  CUCHECK(cudaFree(index));

  int *succ;
  CUCHECK(cudaMalloc((void **)&succ, sizeof(int) * E));
  transform(
      [=] MGPU_DEVICE(int thid) {
        int revEdge = (thid + E / 2) % E;

        if (next[revEdge] == -1) {
          succ[thid] = first[edges_from[revEdge]];
        } else {
          succ[thid] = next[revEdge];
        }
      },
      E, context);
  context.synchronize();

  int *head = new int[1];
  CUCHECK(cudaMemcpy(head, first + root, sizeof(int), cudaMemcpyDeviceToHost));

  CUCHECK(cudaFree(edges_from));
  CUCHECK(cudaFree(next));
  CUCHECK(cudaFree(first));

  cuda_list_rank(E, *head, succ, rank_to_output, context);

  CUCHECK(cudaFree(succ));
}