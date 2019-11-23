#include "bridges/cuda_algorithms.h"
#include "lca/cuda_algorithms.h"

#include<moderngpu/context.hxx>

int main() {
    std::ios_base::sync_with_stdio(false);

    int * dummy;
    mgpu::standard_context_t context(false);
    cuda_bfs(0, 0, dummy, dummy, dummy, dummy, context);
    cuda_lca_naive(0, dummy, 0, dummy, dummy, context);

    return 0;
}
