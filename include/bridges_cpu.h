#ifndef EMC_BRIDGES_CPU_H_
#define EMC_BRIDGES_CPU_H_

void cpu_bridges(int N, int M, const int *row_offsets, const int *col_indices, bool *is_bridge);

void multicore_cpu_bridges(int N, int M, const int *row_offsets, const int *col_indices, bool *is_bridge);

#endif // EMC_BRIDGES_CPU_H_
