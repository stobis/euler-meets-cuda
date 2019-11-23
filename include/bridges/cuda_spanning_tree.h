#ifndef EMC_BRIDGES_CUDA_SPANNING_TREE_H_
#define EMC_BRIDGES_CUDA_SPANNING_TREE_H_

#include <moderngpu/context.hxx>

void spanning_tree(int N, int M, const int* row_offsets, const int* col_indices,
                   bool* is_tree_edge, context_t& context);

#endif  // EMC_BRIDGES_CUDA_SPANNING_TREE_H_
