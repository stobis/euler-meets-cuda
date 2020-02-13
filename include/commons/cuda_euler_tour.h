#ifndef EMC_COMMONS_CUDA_EULER_TOUR_H_
#define EMC_COMMONS_CUDA_EULER_TOUR_H_

#include <moderngpu/context.hxx>

// Input: A tree of N nodes and a list of N-1 edges.
// edges_from[i] and edges_to[i] form an edge.
// root is the only node that should not be on a list of edges_from.
//
// Output: rank[i] is the rank of node i.
void cuda_euler_tour(int N, int root, const int *edges_from,
                     const int *edges_to, int *rank,
                     mgpu::context_t &context);

#endif // EMC_COMMONS_CUDA_EULER_TOUR_H_
