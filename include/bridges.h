#ifndef EMC_BRIDGES_H_
#define EMC_BRIDGES_H_

#include <moderngpu/context.hxx>

namespace emc {

// cuda_bridges_[naive,tarjan,hybrid]:
// Input:   A graph of N nodes and M edges in a csr format (see example below).
//          Input graph should be connected. (You might want to use test/bridges/connect.cpp to select largest component in your graph)
//          All arrays should be on the device.
// Output:  is_bridge[i] is true iff i-th edge (in the order of col_indices) is a bridge.
//
// Example Input:
// N:             4
// M:             8
// row_offsets:   0  2  5  7  8
// col_indices:   1  2  0  2  3  0  1  1
// Result:
// is_bridge:     0  0  0  0  1  0  0  1

void cuda_bridges_naive(int N, int M, const int *row_offsets, const int *col_indices, bool *is_bridge,
                        mgpu::context_t &context);

void cuda_bridges_tarjan(int N, int M, const int *row_offsets, const int *col_indices, bool *is_bridge,
                         mgpu::context_t &context);

void cuda_bridges_hybrid(int N, int M, const int *row_offsets, const int *col_indices, bool *is_bridge,
                         mgpu::context_t &context);

} // namespace emc

#endif // EMC_BRIDGES_H_