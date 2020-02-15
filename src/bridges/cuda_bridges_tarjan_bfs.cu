#include <moderngpu/context.hxx>
#include <moderngpu/kernel_compact.hxx>
#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_segreduce.hxx>
#include <moderngpu/memory.hxx>
using namespace mgpu;

#include <iostream>
#include <utility>
using namespace std;

#include "commons/cuda_euler_tour.h"

#include "bridges/cuda_spanning_tree.h"
#include "bridges/cuda_tarjan_bfs.h"
#include "bridges/gputils.cuh"
#include "bridges/timer.hpp"

namespace {

int segtree_size(int const N) {
    int M = 1;
    while (M < N) {
        M <<= 1;
    }
    return M << 1;
}

template <typename op_t>
void segtree_init(int const N, int const M, int const *init_data, op_t op,
                  int const init_leaf, int *tree, context_t &context) {
    int const FIRST_LEAF_POS = M >> 1;
    dtod(tree + FIRST_LEAF_POS, init_data, N);

    // Fill leafs
    transform([=] MGPU_DEVICE(
                  int index) { tree[FIRST_LEAF_POS + N + index] = init_leaf; },
              M - (FIRST_LEAF_POS + N), context);

    // Fill upper levels
    int begin = FIRST_LEAF_POS >> 1;
    while (begin >= 1) {
        transform(
            [=] MGPU_DEVICE(int index) {
                int const OFFSET = begin + index;
                tree[OFFSET] = op(tree[2 * OFFSET], tree[2 * OFFSET + 1]);
            },
            begin, context);
        begin >>= 1;
    }
}

void spanning_tree(int const N, int const M, int const *row_offsets,
                   int const *row_indices, int const *col_indices,
                   ll *tree_edges, context_t &context) {
    mem_t<bool> is_tree_edge = mgpu::fill<bool>(false, M, context);
    bool *is_tree_edge_data = is_tree_edge.data();

    // spanning_tree(N, M, row_offsets, col_indices, is_tree_edge_data,
    // context); 
    spanning_tree_bfs(N, M, row_offsets, col_indices,
    is_tree_edge_data, context);
    // spanning_tree_ecl(N, M, row_offsets, col_indices, is_tree_edge_data,
    //                   context);

    // Select tree edges
    auto compact = transform_compact(M, context);

    int stream_count = compact.upsweep([=] MGPU_DEVICE(int index) {
        return is_tree_edge_data[index] == true;
    });

    assert(N - 1 == stream_count);

    compact.downsweep([=] MGPU_DEVICE(int dest_index, int source_index) {
        tree_edges[dest_index] =
            (static_cast<ll>(row_indices[source_index]) << 32) |
            (static_cast<ll>(col_indices[source_index]));
    });
}

void list_rank(int const N, int const M, ll const *tree, int *rank,
               context_t &context) {
    mem_t<int> edges_from(M, context);
    mem_t<int> edges_to(M, context);

    transform(
        [=] MGPU_DEVICE(int index, int *edges_from, int *edges_to) {
            edges_from[index] = tree[index] >> 32;
            edges_to[index] = tree[index] & 0xFFFFFFFF;
        },
        M, context, edges_from.data(), edges_to.data());

    cuda_euler_tour(N, 0, edges_from.data(), edges_to.data(), rank, context);
}

void order_by_rank(int const N, int const M, ll *tree, ll *tree_directed,
                   int *tree_directed_rev, int const *rank,
                   context_t &context) {
    transform(
        [=] MGPU_DEVICE(int index) {
            tree_directed[rank[index]] = tree[index];
            tree_directed[rank[index + N - 1]] =
                (static_cast<ll>(tree[index] & 0xFFFFFFFF) << 32) |
                (static_cast<ll>(tree[index]) >> 32);

            tree_directed_rev[rank[index]] = rank[index + N - 1];
            tree_directed_rev[rank[index + N - 1]] = rank[index];
        },
        N - 1, context);
}

void count_preorder(int const N, int const M, ll const *tree_edges,
                    int const *tree_edges_rev, int *preorder,
                    context_t &context) {
    mem_t<int> scan_params(M, context);

    transform(
        [=] MGPU_DEVICE(int index, int *scan_params) {
            scan_params[index] = index < tree_edges_rev[index];
        },
        M, context, scan_params.data());

    scan<scan_type_inc>(scan_params.data(), M, scan_params.data(), context);

    transform(
        [=] MGPU_DEVICE(int index, int const *scan_params) {
            if (index >= tree_edges_rev[index]) return;

            int const to = static_cast<int>(tree_edges[index] & 0xFFFFFFFF);
            preorder[to] = scan_params[index];
        },
        M, context, scan_params.data());
}

template <typename op_t>
void reduce_outgoing(int N, int M, int const *row_offsets,
                     int const *col_indices, int const *row_indices,
                     int const *preorder, int const *parent, op_t op,
                     int op_init, int *output, context_t &context) {
    mem_t<int> reduce_output = mgpu::fill<int>(op_init, N, context);
    transform_segreduce(
        [=] MGPU_DEVICE(int index) {
            int from = preorder[row_indices[index]];
            int to = preorder[col_indices[index]];

            if (to == parent[from]) return op_init;
            return to;
        },
        M, row_offsets, N + 1, reduce_output.data(), op, op_init, context);

    transform(
        [=] MGPU_DEVICE(int index, int const *reduce_output) {
            output[preorder[index]] = reduce_output[index];
        },
        N, context, reduce_output.data());
}

void segtree_query(int const N, int const M, int const *segtree_min,
                   int const *segtree_max, int const *preorder,
                   int const *subtree, bool *is_bridge_end,
                   context_t &context) {
    transform(
        [=] MGPU_DEVICE(int index) {
            // index - vertex id in org graph
            // preorder[index] - preorder id in spanning tree
            // subtree[index] - size of the subtree of index

            int const label = preorder[index];
            int const offset = subtree[index] - 1;

            int va = M / 2 + label;
            int vb = M / 2 + label + offset;

            int mini = min(segtree_min[va], segtree_min[vb]);
            int maxi = max(segtree_max[va], segtree_max[vb]);

            while (va / 2 != vb / 2) {
                if (va % 2 == 0) {
                    mini = min(mini, segtree_min[va + 1]);
                    maxi = max(maxi, segtree_max[va + 1]);
                }
                if (vb % 2 == 1) {
                    mini = min(mini, segtree_min[vb - 1]);
                    maxi = max(maxi, segtree_max[vb - 1]);
                }
                va /= 2;
                vb /= 2;
            }

            if (label <= mini && maxi <= label + offset) {
                is_bridge_end[label] = true;
            }
        },
        N, context);
}

void count_result(int N, int M, int const *row_indices, int const *col_indices,
                  int const *preorder, int const *parent,
                  bool const *is_bridge_end, bool *is_bridge,
                  context_t &context) {
    transform(
        [=] MGPU_DEVICE(int index) {
            int from = preorder[row_indices[index]];
            int to = preorder[col_indices[index]];

            is_bridge[index] = (is_bridge_end[to] && parent[to] == from) ||
                               (is_bridge_end[from] && parent[from] == to);
        },
        M, context);
}

void count_subtree_size_and_parent(int const N, int const M, int *subtree,
                                   int *parent, int const *preorder,
                                   ll const *tree_directed,
                                   int const *tree_directed_rev,
                                   context_t &context) {
    transform(
        [=] MGPU_DEVICE(int index) {
            int const back_index = tree_directed_rev[index];
            if (index >= back_index) return;

            ll const packed = tree_directed[index];
            int const from = static_cast<int>(packed >> 32);
            int const to = static_cast<int>(packed & 0xFFFFFFFF);

            parent[preorder[to]] = preorder[from];
            subtree[to] = (back_index - 1 - index) / 2 + 1;
        },
        M, context);
}

void show_time(string const label, Timer &timer, context_t &context) {
    if (detailed_time) {
        context.synchronize();
        timer.print_and_restart(label);
    }
}

}

void cuda_bridges_tarjan_bfs(int N, int M, int const *row_offsets,
                         int const *col_indices, bool *is_bridge,
                         context_t &context) {
    // Init timer
    Timer timer("tarjan-bfs");

    // Expand csr to edge list
    mem_t<int> row_indices = mgpu::fill<int>(0, M, context);
    _csr_to_list(N, M, row_offsets, row_indices.data(), context);

    // Find spanning tree
    mem_t<ll> tree(N - 1, context);
    spanning_tree(N, M, row_offsets, row_indices.data(), col_indices,
                  tree.data(), context);

    show_time("Spanning Tree", timer, context);

    // List rank
    mem_t<int> rank(2 * (N - 1), context);
    list_rank(N, N - 1, tree.data(), rank.data(), context);

    // Rearrange tree edges using counted ranks
    // Direct edges, find reverse edge index
    mem_t<ll> tree_directed(2 * (N - 1), context);
    mem_t<int> tree_directed_rev(2 * (N - 1), context);
    order_by_rank(N, 2 * (N - 1), tree.data(), tree_directed.data(),
                  tree_directed_rev.data(), rank.data(), context);

    show_time("List rank", timer, context);

    // Count preorder
    mem_t<int> preorder = mgpu::fill(0, N, context);
    count_preorder(N, 2 * (N - 1), tree_directed.data(),
                   tree_directed_rev.data(), preorder.data(), context);

    // Count subtree size & parent
    mem_t<int> subtree = mgpu::fill<int>(N, N, context);
    mem_t<int> parent = mgpu::fill<int>(-1, N, context);
    count_subtree_size_and_parent(N, 2 * (N - 1), subtree.data(), parent.data(),
                                  preorder.data(), tree_directed.data(),
                                  tree_directed_rev.data(), context);
    show_time("Preorder & Subtree size", timer, context);

    // Find local min/max from outgoing edges for every vertex
    mem_t<int> minima(N, context);  // = mgpu::fill<int>(N + 1, N, context);
    mem_t<int> maxima(N, context);  // = mgpu::fill<int>(-1, N, context);

    // Reduce segments to achieve min/max for each
    reduce_outgoing(N, M, row_offsets, col_indices, row_indices.data(),
                    preorder.data(), parent.data(), mgpu::minimum_t<int>(),
                    N + 1, minima.data(), context);
    reduce_outgoing(N, M, row_offsets, col_indices, row_indices.data(),
                    preorder.data(), parent.data(), mgpu::maximum_t<int>(), -1,
                    maxima.data(), context);

    show_time("Local min/max for vertices", timer, context);

    // Segment tree to find min/max for each subtree
    int const SEGTREE_SIZE = segtree_size(N);

    mem_t<int> segtree_min(SEGTREE_SIZE, context);
    segtree_init(N, SEGTREE_SIZE, minima.data(), mgpu::minimum_t<int>(), N + 1,
                 segtree_min.data(), context);

    mem_t<int> segtree_max(SEGTREE_SIZE, context);
    segtree_init(N, SEGTREE_SIZE, maxima.data(), mgpu::maximum_t<int>(), -1,
                 segtree_max.data(), context);

    // Mark bridges ends
    mem_t<bool> is_bridge_end = mgpu::fill<bool>(false, N, context);
    segtree_query(N, SEGTREE_SIZE, segtree_min.data(), segtree_max.data(),
                  preorder.data(), subtree.data(), is_bridge_end.data(),
                  context);

    // Compute result
    count_result(N, M, row_indices.data(), col_indices, preorder.data(),
                 parent.data(), is_bridge_end.data(), is_bridge, context);

    // Print exec time
    show_time("Find bridges", timer, context);
    timer.print_overall();
}
