
#include <moderngpu/context.hxx>
#include <moderngpu/memory.hxx>
using namespace mgpu;

#include "bfs.h"
#include "ecl_cc.cu"
#include "utils.h"

#include "spanning_tree.h"

// This impl use bfs by Adam Polak
void spanning_tree_bfs(int const N, int const M, int const *row_offsets, int const *col_indices,
                       bool *is_tree_edge, context_t &context) {
  mem_t<int> mt_parent = mgpu::fill<int>(-1, N, context);
  mem_t<int> mt_distance = mgpu::fill<int>(-1, N, context);

  bfs(N, M, row_offsets, col_indices, mt_distance.data(), mt_parent.data(), context);

  // Determine tree edges
  mem_t<int> mt_row_indices = mgpu::fill<int>(0, M, context);
  _csr_to_list(N, M, row_offsets, mt_row_indices.data(), context);

  transform(
      [=] MGPU_DEVICE(int index, int const *row_indices, int const *parent) {
        int const from = row_indices[index];
        int const to = col_indices[index];

        if (parent[to] == from) {
          // TODO: this if statement is a fix for multigraph
          if (!(index && row_indices[index - 1] == from && col_indices[index - 1] == to)) {
            is_tree_edge[index] = true;
          }
        }
      },
      M, context, mt_row_indices.data(), mt_parent.data());
}

// This impl use ECL-CC by Texas State University
void spanning_tree_ecl(int const N, int const M, int const *row_offsets, int const *col_indices,
                       bool *is_tree_edge, context_t &context) {
  int *nstat_d;
  if (cudaSuccess != cudaMalloc((void **)&nstat_d, N * sizeof(int))) {
    fprintf(stderr, "ERROR: could not allocate nstat_d,\n\n");
    exit(-1);
  }

  computeCC(N, M, row_offsets, col_indices, nstat_d, is_tree_edge);
}
