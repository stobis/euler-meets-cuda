#ifndef EMC_LCA_H_
#define EMC_LCA_H_

#include <moderngpu/context.hxx>

namespace emc {

// [cuda,cpu]_lca_*:
// Input:   A tree of N nodes, parents[i] is a parent of i-ts node.
//          Q queries, (queries[i*2], queries[i*2+1]) is a i-th query
//          In cuda_lca* all arrays should be on the device.
// Output:  answers[i] is a lowest common ancestor of queries[i*2] and queries[i*2+1]
//
// Example Input:
// N:           6
// parents:    -1  0  0  1  1  3
// Q:           3
// queries:     3  5  0  2  4  5
// Result:
// answers:     3  0  1
void cuda_lca_inlabel(int N, const int *parents, int Q, const int *queries, int *answers, int batchSize,
                      mgpu::context_t &context);

void cuda_lca_naive(int N, const int *parents, int Q, const int *queries, int *answers, int batchSize,
                    mgpu::context_t &context);

void cpu_lca_simple(int N, const int *parents, int Q, const int *queries, int *answers);

void cpu_lca_rmq(int N, const int *parents, int Q, const int *queries, int *answers);

void cpu_lca_inlabel(int N, const int *parents, int Q, const int *queries, int *answers);

} // namespace emc

#endif // EMC_LCA_H_