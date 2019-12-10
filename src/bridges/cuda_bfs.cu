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
#include "bridges/graph.hpp"
#include "bridges/test-result.hpp"
#include "bridges/timer.hpp"

void print(mem_t<int>& vec) {
    auto hvec = from_mem(vec);
    for (auto x : hvec) cout << x << " ";
    cout << endl;
}

void _csr_to_list(int N, int M, const int* row_offsets, int* row_indices,
                  mgpu::context_t& context) {
    transform(
        [] MGPU_DEVICE(int index, int const* row_offsets, int* row_indices) {
            if (index == 0) return;
            row_indices[row_offsets[index]] = 1;
        },
        N, context, row_offsets, row_indices);

    scan<scan_type_inc>(row_indices, M, row_indices, context);
}

void _mark_bridges(int N, int M, const int* row_offsets, const int* col_indices,
                   const int* distance, const int* parent, bool* is_bridge,
                   context_t& context) {
    mem_t<int> marked = mgpu::fill<int>(0, N, context);
    mem_t<int> row_indices = mgpu::fill<int>(0, M, context);

    _csr_to_list(N, M, row_offsets, row_indices.data(), context);

    // Mark nodes visited during traversal
    transform(
        [] MGPU_DEVICE(int index, int const* row_indices,
                       int const* col_indices, int const* distance,
                       int const* parent, int* marked) {
            int const from = row_indices[index];
            int const to = col_indices[index];

            // Check if tree edge
            if (parent[to] == from || parent[from] == to) {
                return;
            }

            int higher = distance[to] < distance[from] ? to : from;
            int lower = higher == to ? from : to;
            int diff = distance[lower] - distance[higher];

            // Equalize heights
            while (diff--) {
                marked[lower] = 1;
                lower = parent[lower];
            }

            // Mark till LCA is found
            while (lower != higher) {
                marked[lower] = 1;
                lower = parent[lower];

                marked[higher] = 1;
                higher = parent[higher];
            }
        },
        M, context, row_indices.data(), col_indices, distance, parent,
        marked.data());

    // Fill result array
    transform(
        [] MGPU_DEVICE(int index, int const* row_indices,
                       int const* col_indices, int const* parent,
                       int const* marked, bool* is_bridge) {
            int const to = row_indices[index];
            int const from = col_indices[index];

            is_bridge[index] = (parent[to] == from && marked[to] == 0) ||
                               (parent[from] == to && marked[from] == 0);
        },
        M, context, row_indices.data(), col_indices, parent, marked.data(),
        is_bridge);
}

void cuda_bfs(int N, int M, const int* row_offsets, const int* col_indices,
              int* distance, int* parent, mgpu::context_t& context) {
    // Execute BFS to compute distances (needed for determine parents)
    // TODO: rm unnecessary dtod copy
    mem_t<int> mt_distance = mgpu::fill<int>(-1, N, context);
    mem_t<int> mt_row_offsets(N+1, context);
    mem_t<int> mt_col_indices(M, context);
    dtod(mt_row_offsets.data(), row_offsets, N+1);
    dtod(mt_col_indices.data(), col_indices, M);
    bfs_mgpu::ParallelBFS(N, M, mt_row_offsets, mt_col_indices, 1, mt_distance,
                          context);
    dtod(distance, mt_distance.data(), N);

    // Determine parents
    mem_t<int> mt_row_indices = mgpu::fill<int>(0, M, context);
    _csr_to_list(N, M, row_offsets, mt_row_indices.data(), context);

    transform(
        [] MGPU_DEVICE(int index, int const* row_indices,
                       int const* col_indices, int const* distance,
                       int* parent) {
            int const from = row_indices[index];
            int const to = col_indices[index];

            if (distance[from] == distance[to] - 1) {
                parent[to] = from;
            }
        },
        M, context, mt_row_indices.data(), col_indices, distance, parent);
}

void cuda_bridges_naive(int N, int M, const int* row_offsets,
                        const int* col_indices, bool* is_bridge,
                        context_t& context) {
    // Init timer
    Timer timer("gpu-bfs");

    // Allocate array(s)
    mem_t<int> distance = mgpu::fill<int>(-1, N, context);
    mem_t<int> parent = mgpu::fill<int>(-1, N, context);

    // Execute BFS to compute distances (needed for determine parents)
    // bfs_mgpu::ParallelBFS(n, directed_m, nodes, edges_to, 0, distance,
    // context);
    cuda_bfs(N, M, row_offsets, col_indices, distance.data(), parent.data(),
             context);

    if (detailed_time) {
        context.synchronize();
        timer.print_and_restart("BFS");
    }

    // Find bridges
    _mark_bridges(N, M, row_offsets, col_indices, distance.data(),
                  parent.data(), is_bridge, context);

    if (detailed_time) {
        context.synchronize();
        timer.print_and_restart("Find Bridges");
        timer.print_overall();
    }

    if (detailed_time) {
        mem_t<int> maxd(1, context);
        reduce(distance.data(), distance.size(), maxd.data(),
               mgpu::maximum_t<int>(), context);
        vector<int> maxd_host = from_mem(maxd);

        context.synchronize();
        timer.print_and_restart("Max distance: " +
                                to_string(maxd_host.front()));
    }
}
