#include <iostream>

#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/transform.hxx>

#include "lca/cuda_commons.h"
#include "lca/cuda_euler_tour.h"
#include "lca/cuda_list_rank.h"

using namespace std;
using namespace mgpu;

void cuda_euler_tour(int N, int root, const int *edges_from_input,
                     const int *edges_to_input, int *rank,
                     mgpu::context_t &context) {

  int E = 2 * N - 2;

  int *edges_to;
  CUCHECK(cudaMalloc((void **)&edges_from_to_scan, sizeof(int) * (E)));
  int *edges_from;
  CUCHECK(cudaMalloc((void **)&edges_from, sizeof(int) * (E)));
  int *index;
  CUCHECK(cudaMalloc((void **)&index, sizeof(int) * (E)));

  transform(
      [=] MGPU_DEVICE(int thid) {
        edges_from[thid + E / 2] = edges_to[thid] = edges_to_input[thid];
        edges_to[thid + E / 2] = edges_from[thid] = edges_from_input[thid];
        index[thid] = thid;
      },
      N, context);

  mergesort(index, E,
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

  int *rev_index;
  CUCHECK(cudaMalloc((void **)&rev_index, sizeof(int) * E));

  transform([=] MGPU_DEVICE(int thid) { rev_index[index[thid]] = thid; }, E,
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
          first[t] = thid;
          return;
        }

        int pf = edges_from[index[thid - 1]];
        int pt = edges_to[index[thid - 1]];

        if (f != pf) {
          first[f] = thid;
        } else {
          next[thid - 1] = thid;
        }
      },
      E, context);

  int *succ;
  CUCHECK(cudaMalloc((void **)&succ, sizeof(int) * E));
  transform(
      [=] MGPU_DEVICE(int thid) {
        if (next[thid] == -1) {
          int to = edges_from[rev_index[thid]];
          succ[thid] = first[to];
        } else {
          int reverse_edge = (rev_index[thid] + E / 2) % E;
          succ[thid] = next[reverse_edge];
        }
      },
      E, context);

  cuda_list_rank(E, first[root], succ, rank, context);

  context.synchronize();

  cerr << "Rank: " << endl;
  CudaPrintTab(rank, N * 2 - 2);
}