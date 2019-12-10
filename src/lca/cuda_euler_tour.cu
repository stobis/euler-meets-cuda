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
                     const int *edges_to_input, int *rank_to_output,
                     mgpu::context_t &context)
{

  int E = N * 2 - 2;

  int *edges_to;
  CUCHECK(cudaMalloc((void **)&edges_to, sizeof(int) * E));
  int *edges_from;
  CUCHECK(cudaMalloc((void **)&edges_from, sizeof(int) * E));
  int *index;
  CUCHECK(cudaMalloc((void **)&index, sizeof(int) * E));

  transform(
      [=] MGPU_DEVICE(int thid) {
        if (thid == root)
          return;

        int afterRoot = 0;
        if (thid > root)
          afterRoot = 1;

        edges_from[thid + N - afterRoot - 1] = edges_to[thid - afterRoot] =
            edges_to_input[thid];
        edges_to[thid + N - afterRoot - 1] = edges_from[thid - afterRoot] =
            edges_from_input[thid];
      },
      N, context);

  transform([=] MGPU_DEVICE(int thid) { index[thid] = thid; }, E, context);

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
  CUCHECK(cudaMalloc((void **)&rev_index, sizeof(int) * (E)));

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

        if (thid == 0)
        {
          first[f] = index[thid];
          // printf("thid:%d, f:%d, t:%d, first[%d]:%d\n", thid, f, t, f, thid);
          return;
        }

        int pf = edges_from[index[thid - 1]];
        int pt = edges_to[index[thid - 1]];

        if (f != pf)
        {
          first[f] = index[thid];
          // printf("thid:%d, f:%d, t:%d, pf:%d, pt:%d, first[%d]:%d\n",
          //  thid, f, t, pf, pt, f, index[thid]);
        }
        else
        {
          next[index[thid - 1]] = index[thid];
          // printf("thid:%d, f:%d, t:%d, pf:%d, pt:%d, next[%d]:%d\n",
          //  thid, f, t, pf, pt, index[thid - 1], index[thid]);
        }
      },
      E, context);

  // cout << "From: " << endl;
  // CudaPrintTab(edges_from, E);
  // cout << "To:" << endl;
  // CudaPrintTab(edges_to, E);
  // cout << "Index:" << endl;
  // CudaPrintTab(index, E);
  // cout << "Rev_index:" << endl;
  // CudaPrintTab(rev_index, E);
  // cout << "Next:" << endl;
  // CudaPrintTab(next, E);
  // cout << "First:" << endl;
  // CudaPrintTab(first, N);

  int *succ;
  CUCHECK(cudaMalloc((void **)&succ, sizeof(int) * E));
  transform(
      [=] MGPU_DEVICE(int thid) {
        int revEdge = (thid + E / 2) % E;

        if (next[revEdge] == -1)
        {
          succ[thid] = first[edges_from[revEdge]];
        }
        else
        {
          succ[thid] = next[revEdge];
        }
      },
      E, context);

  // cout << "Succ:" << endl;
  // CudaPrintTab(succ, E);

  int *head = new int[1];
  CUCHECK(cudaMemcpy(head, first + root, sizeof(int), cudaMemcpyDeviceToHost));

  int *rank;
  CUCHECK(cudaMalloc((void **)&rank, sizeof(int) * E));

  cuda_list_rank(E, *head, succ, rank, context);

  // cout << "Rank: " << endl;
  // CudaPrintTab(rank, E);

  transform([=] MGPU_DEVICE(int thid) {
    if (thid == root || thid == root + N)
    {
      rank_to_output[thid] = -1;
      return;
    }
    int rank_addr = thid;
    if (thid > root)
      rank_addr--;
    if (thid > root + N)
      rank_addr--;
    rank_to_output[thid] = rank[rank_addr];
  },
            E + 2, context);

  // cout << "Rank output: " << endl;
  // CudaPrintTab(rank_to_output, E + 2);
}