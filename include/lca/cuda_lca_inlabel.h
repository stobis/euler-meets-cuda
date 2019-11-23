#ifndef EMC_LCA_CUDA_LCA_INLABEL_H_
#define EMC_LCA_CUDA_LCA_INLABEL_H_

#include <moderngpu/context.hxx>

void cuda_lca_inlabel(int N, const int* parents, int Q,
                               const int* queries, int* answers,
                               mgpu::context_t& context);

#endif  // EMC_LCA_CUDA_LCA_INLABEL_H_
