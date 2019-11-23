#include "lca/cuda_lca_naive.h"

#include <iostream>
#include <moderngpu/context.hxx>

void cuda_lca_naive(int N, const int* parents, int Q, const int* queries,
                    int* answers, mgpu::context_t& context) {
    std::cout << "inside cuda_lca_naive" << std::endl;
    return;
}
