#ifndef EMC_BFS_H_
#define EMC_BFS_H_

#include <moderngpu/context.hxx>
using namespace mgpu;

void bfs(const int *nodes, int n, const int *edges, int m, int start, int *distance, context_t &context);

void bfs(int N, int M, const int *row_offsets, const int *col_indices, int *distance, int *parent,
         context_t &context);

#endif // EMC_BFS_H_
