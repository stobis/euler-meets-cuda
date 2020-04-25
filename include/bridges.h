#ifndef EMC_BRIDGES_H_
#define EMC_BRIDGES_H_

#include <moderngpu/context.hxx>

namespace emc {

void cuda_bridges_naive(int N, int M, const int *row_offsets, const int *col_indices, bool *is_bridge,
                        mgpu::context_t &context);

void cuda_bridges_tarjan(int N, int M, const int *row_offsets, const int *col_indices, bool *is_bridge,
                         mgpu::context_t &context);

void cuda_bridges_hybrid(int N, int M, const int *row_offsets, const int *col_indices, bool *is_bridge,
                         mgpu::context_t &context);

} // namespace

#endif // EMC_BRIDGES_H_
