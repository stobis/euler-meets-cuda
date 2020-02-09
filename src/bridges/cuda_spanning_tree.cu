#include <moderngpu/context.hxx>
#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/memory.hxx>
using namespace mgpu;

#include <iostream>
#include <utility>
using namespace std;

#include "conn.cuh"

#include "lca/cuda_euler_tour.h"

#include "bridges/cuda_spanning_tree.h"
#include "bridges/gputils.cuh"
#include "bridges/timer.hpp"

// This impl uses cc by Soman & Kothapalli
void spanning_tree(int const N, int const M, int const *row_indices, int const *col_indices,
                   bool *is_tree_edge, context_t &context)
{
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
