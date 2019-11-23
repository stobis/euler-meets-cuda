#ifndef EMC_BRIDGES_CUDA_TARJAN_H_
#define EMC_BRIDGES_CUDA_TARJAN_H_

#include <moderngpu/context.hxx>

void cuda_bridges_tarjan(int N, int M, const int* row_offsets,
                         const int* col_indices, bool* is_bridge,
                         mgpu::context_t& context);

#endif  // EMC_BRIDGES_CUDA_TARJAN_H_
