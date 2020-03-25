
#ifndef EMC_LCA_H_
#define EMC_LCA_H_

#include "commons/commons.h"
#include <moderngpu/context.hxx>

// [cuda,cpu]_lca_*:
// Input:   A tree of N nodes, parents[i] is a parent of i-ts node.
//          Q queries, (queries[i*2], queries[i*2+1]) is a i-th query
//          In cuda_lca* all arrays should be on the device.
// Output:  answers[i] is a lowest common ancestor of queries[i*2] and queries[i*2+1]
//
// Example Input:
// N:           6
// parents:    -1  0  0  1  1  3
// Q:           3
// queries:     3  5  0  2  4  5
// Result:
// answers:     3  0  1
void cuda_lca_inlabel(int N, const int *parents, int Q, const int *queries, int *answers,
                      mgpu::context_t &context);

void cuda_lca_naive(int N, const int *parents, int Q, const int *queries, int *answers,
                    mgpu::context_t &context);

void cpu_lca_simple(int N, const int *parents, int Q, const int *queries, int *answers);

void cpu_lca_rmq(int N, const int *parents, int Q, const int *queries, int *answers);

void cpu_lca_inlabel(int N, const int *parents, int Q, const int *queries, int *answers);

// cuda_euler_tour:
// Input:  A tree of N nodes and a list of N-1 edges.
//         edges_from[i] and edges_to[i] form an edge.
//         All arrays should be on the device.
// Output: rank[i] is the rank of edge edges_from[i] --> edged_to[i]
//         rank[i + N-1] is the rank of edge edges_to[i] --> edges_from[i]
//
// Example Input:
// N:           5
// root:        2
// edges_from:  0  1  3  4
// edges_to:    1  2  1  0
// Result:
// rank:        6  7  2  5  3  0  1  4
void cuda_euler_tour(int N, int root, const int *edges_from, const int *edges_to, int *rank,
                     mgpu::context_t &context);

void cuda_list_rank(int N, int head, const int *next, int *rank, mgpu::context_t &context);

#endif // EMC_LCA_H_