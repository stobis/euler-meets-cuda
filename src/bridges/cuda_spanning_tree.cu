#include <moderngpu/context.hxx>
#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/memory.hxx>
using namespace mgpu;

#include <iostream>
#include <utility>
using namespace std;

#include "ECL-CC_10_mod.cu"
#include "bridges/cuda_bfs.h"
#include "conn.cuh"

#include "bridges/cuda_spanning_tree.h"
#include "bridges/gputils.cuh"
#include "bridges/timer.hpp"

// This impl use cc by Soman & Kothapalli
void spanning_tree(int const N, int const M, int const *row_offsets,
                   int const *col_indices, bool *is_tree_edge,
                   context_t &context) {
    mem_t<int> mt_row_indices = mgpu::fill<int>(0, M, context);
    _csr_to_list(N, M, row_offsets, mt_row_indices.data(), context);
    int const * row_indices = mt_row_indices.data();

    mem_t<cc::edge> device_cc_graph(M, context);
    cc::edge *device_cc_graph_data = device_cc_graph.data();

    mem_t<int> mapping = mgpu::fill<int>(-1, M / 2, context);
    int *mapping_data = mapping.data();

    mem_t<int> device_tree_edges = mgpu::fill<int>(0, M / 2, context);
    int *device_tree_edges_data = device_tree_edges.data();

    // Construct input for CC
    auto compact = transform_compact(M, context);

    int stream_count = compact.upsweep([=] MGPU_DEVICE(int index) {
        return row_indices[index] < col_indices[index];
        // TODO: consider hash(r, c) < hash(c, r)
    });

    compact.downsweep([=] MGPU_DEVICE(int dest_index, int source_index) {
        device_cc_graph_data[dest_index].x =
            (static_cast<ll>(row_indices[source_index]) << 32) |
            (static_cast<ll>(col_indices[source_index]));
        mapping_data[dest_index] = source_index;
    });
    assert(M / 2 == stream_count);

    // Use CC algorithm to find spanning tree
    cc::compute(N, M / 2, device_cc_graph_data, device_tree_edges_data);

    // Set indicators in output array
    transform(
        [=] MGPU_DEVICE(int index) {
            is_tree_edge[mapping_data[index]] =
                device_tree_edges_data[index] ? true : false;
        },
        M / 2, context);
}

// This impl use bfs by Adam Polak
void spanning_tree_bfs(int const N, int const M, int const *row_offsets,
                       int const *col_indices, bool *is_tree_edge,
                       context_t &context) {
    mem_t<int> mt_parent = mgpu::fill<int>(-1, N, context);
    mem_t<int> mt_distance = mgpu::fill<int>(-1, N, context);

    cuda_bfs(N, M, row_offsets, col_indices, mt_distance.data(), mt_parent.data(), context);

    // Determine tree edges
    mem_t<int> mt_row_indices = mgpu::fill<int>(0, M, context);
    _csr_to_list(N, M, row_offsets, mt_row_indices.data(), context);

    transform(
        [=] MGPU_DEVICE(int index, int const *row_indices, int const *parent) {
            int const from = row_indices[index];
            int const to = col_indices[index];

            if (parent[to] == from) {
                // TODO: this if statement is a fix for multigraph
                if (!(index && row_indices[index-1] == from && col_indices[index-1] == to)) {
                    is_tree_edge[index] = true;
                }
            }
        },
        M, context, mt_row_indices.data(), mt_parent.data());
}

// This impl use ECL-CC by Texas State University
void spanning_tree_ecl(int const N, int const M, int const *row_offsets,
                       int const *col_indices, bool *is_tree_edge,
                       context_t &context) {
    int* nstat_d;
    if (cudaSuccess != cudaMalloc((void **)&nstat_d, N * sizeof(int))) {fprintf(stderr, "ERROR: could not allocate nstat_d,\n\n");  exit(-1);}
    
    computeCC(N, M, row_offsets, col_indices, nstat_d, is_tree_edge);
}
