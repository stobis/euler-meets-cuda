
#include <moderngpu/context.hxx>
#include <moderngpu/kernel_load_balance.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/memory.hxx>
using namespace mgpu;

#include "utils.h"

#include "bfs.h"

// BFS by Adam Polak
void bfs(const int *nodes, int n, const int *edges, int m, int start, int *distance, context_t &context) {
  mem_t<int> node_frontier(n, context);
  mem_t<int> edge_frontier(m, context);
  mem_t<int> segments(max(n, m), context);
  mem_t<int> count(1, context);
  int *node_frontier_ptr = node_frontier.data();

  cudaMemset(distance, -1, n * sizeof(int));
  cudaMemset(distance + start, 0, sizeof(int));
  htod(node_frontier.data(), &start, 1);
  int node_frontier_size = 1;

  int level = 0;

  while (node_frontier_size > 0) {
    ++level;
    transform_scan<int>(
        [=] MGPU_DEVICE(int i) {
          int v = node_frontier_ptr[i];
          return nodes[v + 1] - nodes[v];
        },
        node_frontier_size, segments.data(), plus_t<int>(), count.data(), context);

    int edge_frontier_size = from_mem(count)[0];

    transform_lbs(
        [] MGPU_DEVICE(int i, int seg, int rank, const int *edges, const int *nodes, const int *node_frontier,
                       int *edge_frontier) { edge_frontier[i] = edges[nodes[node_frontier[seg]] + rank]; },
        edge_frontier_size, segments.data(), node_frontier_size, context, edges, nodes, node_frontier.data(),
        edge_frontier.data());

    transform(
        [] MGPU_DEVICE(int i, int level, const int *edge_frontier, int *distance, int *segments) {
          segments[i] = (-1 == atomicCAS(distance + edge_frontier[i], -1, level));
        },
        edge_frontier_size, context, level, edge_frontier.data(), distance, segments.data());

    scan(segments.data(), edge_frontier_size, segments.data(), plus_t<int>(), count.data(), context);

    node_frontier_size = from_mem(count)[0];

    transform_lbs(
        [] MGPU_DEVICE(int i, int seg, int rank, const int *edge_frontier, int *node_frontier) {
          assert(rank == 0);
          node_frontier[i] = edge_frontier[seg];
        },
        node_frontier_size, segments.data(), edge_frontier_size, context, edge_frontier.data(),
        node_frontier.data());
  }
}

// BFS + parents
void bfs(int N, int M, const int *row_offsets, const int *col_indices, int *distance, int *parent,
         mgpu::context_t &context) {
  // Execute BFS to compute distances (needed for determine parents)
  bfs(row_offsets, N, col_indices, M, 0, distance, context);

  // Determine parents
  mem_t<int> mt_row_indices = mgpu::fill<int>(0, M, context);
  _csr_to_list(N, M, row_offsets, mt_row_indices.data(), context);

  transform(
      [=] MGPU_DEVICE(int index, int const *row_indices) {
        int const from = row_indices[index];
        int const to = col_indices[index];

        if (distance[from] == distance[to] - 1) {
          parent[to] = from;
        }
      },
      M, context, mt_row_indices.data());
}
