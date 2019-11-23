#ifndef EMC_LCA_CUDA_EULER_TOUR_H_
#define EMC_LCA_CUDA_EULER_TOUR_H_

#include <moderngpu/context.hxx>

void cuda_euler_tour(int N, int root, const int* row_offsets,
                     const int* col_indices, int* rank, context_t& context);

#endif  // EMC_LCA_CUDA_EULER_TOUR_H_
