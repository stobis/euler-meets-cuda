#include <iostream>

#include "bridges/cuda_algorithms.h"

#include <moderngpu/context.hxx>
#include <moderngpu/memory.hxx>

void print_ans(int const M, bool *is_bridge)
{
  bool *ib = new bool[M];
  mgpu::dtoh(ib, is_bridge, M);
  for (int i = 0; i < M; ++i)
    std::cout << ib[i] << " ";
  std::cout << std::endl;
}

int main(int argc, char *argv[])
{
  std::ios_base::sync_with_stdio(false);

  
    int *dummy;
    bool *dummy2;
    mgpu::standard_context_t context(false);

    // Minimal example
    int const N = 5;
    int const M = 10;
    std::vector<int> ro{0, 3, 5, 6, 8, 10};
    std::vector<int> ci{1, 4, 3, 0, 4, 3, 2, 0, 0, 1};
    mgpu::mem_t<int> row_offsets = mgpu::to_mem(ro, context);
    mgpu::mem_t<int> col_indices = mgpu::to_mem(ci, context);
    mgpu::mem_t<bool> is_bridge = mgpu::fill<bool>(false, M, context);

    cuda_bridges_naive(N, M, row_offsets.data(), col_indices.data(), is_bridge.data(), context);
    print_ans(M, is_bridge.data());
    is_bridge = mgpu::fill<bool>(false, M, context);
    cuda_bridges_tarjan(N, M, row_offsets.data(), col_indices.data(), is_bridge.data(), context);
    print_ans(M, is_bridge.data());
    // cuda_lca_naive(0, dummy, 0, dummy, dummy, context);

    // bool * ib = new bool[M];
    // mgpu::dtoh(ib, is_bridge.data(), M);
    // for (int i = 0; i < M; ++i) std::cout << ib[i] << " ";
    // std::cout << std::endl;

  return 0;
}
