#ifndef EMC_EULER_H_
#define EMC_EULER_H_

#include <moderngpu/context.hxx>

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

#endif // EMC_EULER_H_