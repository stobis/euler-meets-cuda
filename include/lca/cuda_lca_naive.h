#ifndef EMC_LCA_CUDA_LCA_NAIVE_H_
#define EMC_LCA_CUDA_LCA_NAIVE_H_

#include <moderngpu/context.hxx>

void cuda_lca_naive(int N, const int* parents, int Q, const int* queries,
                    int* answers, context_t& context);

#endif  // EMC_LCA_CUDA_LCA_NAIVE_H_
