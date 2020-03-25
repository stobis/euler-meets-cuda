
#ifndef EMC_LCA_H_
#define EMC_LCA_H_

#include "commons/commons.h"
#include <moderngpu/context.hxx>

void cuda_lca_inlabel(int N, const int *parents, int Q, const int *queries,
                      int *answers, mgpu::context_t &context);

void cuda_lca_naive(int N, const int *parents, int Q, const int *queries,
                    int *answers, mgpu::context_t &context);

void cpu_lca_simple(int N, const int *parents, int Q, const int *queries,
                    int *answers);

void cpu_lca_rmq(int N, const int *parents, int Q, const int *queries,
                 int *answers);

void cpu_lca_inlabel(int N, const int *parents, int Q, const int *queries,
                     int *answers);

#endif // EMC_LCA_H_