#ifndef EMC_BRIDGES_CUDA_HYBRID_H_
#define EMC_BRIDGES_CUDA_HYBRID_H_

#include <moderngpu/context.hxx>

void cuda_bridges_hybrid(int N, int M, const int* row_offsets,
                         const int* col_indices, bool* is_bridge,
                         mgpu::context_t& context);

#endif  // EMC_BRIDGES_CUDA_HYBRID_H_
