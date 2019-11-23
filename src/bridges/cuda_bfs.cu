#include "bridges/cuda_bfs.h"

#include <iostream>
#include <moderngpu/context.hxx>

void cuda_bfs(int N, int M, const int* row_offsets, const int* col_indices,
              int* distance, int* parent, mgpu::context_t& context) {
    std::cout << "inside cuda_bfs" << std::endl;
    return;
}
