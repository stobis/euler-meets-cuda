#ifndef EMC_COMMONS_CUDA_LIST_RANK_H_
#define EMC_COMMONS_CUDA_LIST_RANK_H_

#include <moderngpu/context.hxx>

void cuda_list_rank(int N, int head, const int* next, int* rank,
                    mgpu::context_t& context);

#endif  // EMC_COMMONS_CUDA_LIST_RANK_H_
