#ifndef EMC_BRIDGES_CUDA_BFS_H_
#define EMC_BRIDGES_CUDA_BFS_H_

#include <moderngpu/context.hxx>

void cuda_bfs(int N, int M, const int* row_offsets, const int* col_indices,
              int* distance, int* parent, context_t& context);

#endif  // EMC_BRIDGES_CUDA_BFS_H_
