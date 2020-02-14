#include <moderngpu/context.hxx>
#include <moderngpu/kernel_reduce.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/memory.hxx>
using namespace mgpu;

#include <iostream>
#include <vector>
using namespace std;

#include "bfs-mgpu.cuh"
#include "bridges/cuda_bfs.h"
#include "bridges/gputils.cuh"

void cuda_bfs(int N, int M, const int* row_offsets, const int* col_indices,
              int* distance, int* parent, mgpu::context_t& context) {
    // Execute BFS to compute distances (needed for determine parents)
    bfs_mgpu::ParallelBFS(N, M, row_offsets, col_indices, 0, distance, context);

    // Determine parents
    mem_t<int> mt_row_indices = mgpu::fill<int>(0, M, context);
    _csr_to_list(N, M, row_offsets, mt_row_indices.data(), context);

    transform(
        [=] MGPU_DEVICE(int index, int const* row_indices) {
            int const from = row_indices[index];
            int const to = col_indices[index];

            if (distance[from] == distance[to] - 1) {
                parent[to] = from;
            }
        },
        M, context, mt_row_indices.data());
}
