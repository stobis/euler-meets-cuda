#ifndef EMC_SPANNING_TREE_H_
#define EMC_SPANNING_TREE_H_

#include <moderngpu/context.hxx>

void spanning_tree_bfs(int N, int M, const int *row_offsets, const int *col_indices, bool *is_tree_edge,
                       mgpu::context_t &context);

void spanning_tree_ecl(int N, int M, const int *row_offsets, const int *col_indices, bool *is_tree_edge,
                       mgpu::context_t &context);

#endif // EMC_SPANNING_TREE_H_
