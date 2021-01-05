#include "bridges_cpu.h"

#include "timer.hpp"

#include <string.h>

#include <memory>
#include <vector>

#include <parallel/numeric>

using namespace std;

// MULTICORE CPU

typedef struct {
  int n;
  int m;
  const int* out_array;
  const int* out_degree_list;
} graph;
#define out_degree(g, n) (g->out_degree_list[n+1] - g->out_degree_list[n])
#define out_vertices(g, n) (&g->out_array[g->out_degree_list[n]])

void bcc_bfs_do_bfs(graph* g, int root, int* parents, int* levels);

void multicore_cpu_bridges(int N, int M, const int *row_offsets, const int *col_indices, bool *is_bridge) {
  Timer timer("multicore");

  graph g{N, M, col_indices, row_offsets};
  unique_ptr<int[]> parents(new int[N]), levels(new int[N]);
  memset(parents.get(), -1, N * sizeof(int));
  memset(levels.get(), -1, N * sizeof(int));
  bcc_bfs_do_bfs(&g, 0, parents.get(), levels.get());

  timer.print_and_restart("BFS");

  unique_ptr<int[]> row_indices(new int[M]);
  memset(row_indices.get(), 0, M * sizeof(int));
  #pragma omp parallel for
  for (int i = 1; i < N; ++i)
    row_indices[row_offsets[i]] = 1;
  __gnu_parallel::partial_sum(row_indices.get(), row_indices.get() + M, row_indices.get());

  unique_ptr<bool[]> marked(new bool[N]);
  memset(marked.get(), 0, sizeof(bool) * N);

  #pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    int a = row_indices[i], b = col_indices[i];
    if (a < b || parents[a] == b || parents[b] == a)
      continue;
    if (levels[a] < levels[b])
      swap(a, b);
    int diff = levels[a] - levels[b];
    while (diff--) {
      marked[a] = true;
      a = parents[a];
    }
    while (a != b) {
      marked[a] = marked[b] = true;
      a = parents[a];
      b = parents[b];
    }
  }

  #pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    int a = row_indices[i], b = col_indices[i];
    is_bridge[i] = (parents[a] == b && !marked[a]) || (parents[b] == a && !marked[b]);
  }

  timer.print_and_restart("Marking");
  timer.print_overall();
}

/* 
 * The multicore BFS implementation below comes from 
 *
 * https://github.com/HPCGraphAnalysis/bicc/tree/master/bcc-hipc14
 * 
 * G. M. Slota and K. Madduri,
 * Simple Biconnectivity Algorithms for Multicore Platforms,
 * in the Proceedings of the IEEE Conference on High Performance Computing
 * (HiPC), 2014.
 *
 * Copyright (c) 2014, The Pennsylvania State University.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * 
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of The Pennsylvania State University nor the 
 *    names of its contributors may be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#define THREAD_QUEUE_SIZE 1024

inline void empty_queue(int* thread_queue, int& thread_queue_size, int* queue_next, int& queue_size_next) {
  int start_offset;
  #pragma omp atomic capture
  start_offset = queue_size_next += thread_queue_size;

  start_offset -= thread_queue_size;
  for (int i = 0; i < thread_queue_size; ++i)
    queue_next[start_offset + i] = thread_queue[i];
  thread_queue_size = 0;
}

inline void add_to_queue(int* thread_queue, int& thread_queue_size, int* queue_next, int& queue_size_next, int vert) {
  thread_queue[thread_queue_size++] = vert;
  if (thread_queue_size == THREAD_QUEUE_SIZE)
    empty_queue(thread_queue, thread_queue_size, queue_next, queue_size_next);
}

#define ALPHA 15.0
#define BETA 24.0

void bcc_bfs_do_bfs(graph* g, int root, int* parents, int* levels)
{
  int num_verts = g->n;
  double avg_out_degree = g->m / (double)g->n;

  int* queue = new int[num_verts];
  int* queue_next = new int[num_verts];
  int queue_size = 0;  
  int queue_size_next = 0;

  queue[0] = root;
  queue_size = 1;
  parents[root] = root;
  levels[root] = 0;

  int level = 1;
  int num_descs = 0;
  int local_num_descs = 0;
  bool use_hybrid = false;
  bool already_switched = false;

  #pragma omp parallel
  {
    int thread_queue[ THREAD_QUEUE_SIZE ];
    int thread_queue_size = 0;

    while (queue_size) {
      if (!use_hybrid) {
        #pragma omp for schedule(guided) reduction(+:local_num_descs) nowait
        for (int i = 0; i < queue_size; ++i) {
          int vert = queue[i];
          int out_degree = out_degree(g, vert);
          const int* outs = out_vertices(g, vert);
          for (int j = 0; j < out_degree; ++j) {      
            int out = outs[j];
            if (levels[out] < 0) {
              levels[out] = level;
              parents[out] = vert;
              ++local_num_descs;
              add_to_queue(thread_queue, thread_queue_size, queue_next, queue_size_next, out);
            }
          }
        }
      } else {
        int prev_level = level - 1;
        #pragma omp for schedule(guided) reduction(+:local_num_descs) nowait
        for (int vert = 0; vert < num_verts; ++vert) {
          if (levels[vert] < 0) {
            int out_degree = out_degree(g, vert);
            const int* outs = out_vertices(g, vert);
            for (int j = 0; j < out_degree; ++j) {
              int out = outs[j];
              if (levels[out] == prev_level) {
                levels[vert] = level;
                parents[vert] = out;
                ++local_num_descs;
                add_to_queue(thread_queue, thread_queue_size, queue_next, queue_size_next, vert);
                break;
              }
            }
          }
        }
      }
      
      empty_queue(thread_queue, thread_queue_size, queue_next, queue_size_next);
      #pragma omp barrier

      #pragma omp single
      {
        num_descs += local_num_descs;

        if (!use_hybrid) {  
          double edges_frontier = (double)local_num_descs * avg_out_degree;
          double edges_remainder = (double)(num_verts - num_descs) * avg_out_degree;
          if ((edges_remainder / ALPHA) < edges_frontier && edges_remainder > 0 && !already_switched)
            use_hybrid = true;
        } else {
          if (((double)num_verts / BETA) > local_num_descs  && !already_switched) {
            use_hybrid = false;
            already_switched = true;
          }
        }
        local_num_descs = 0;

        queue_size = queue_size_next;
        queue_size_next = 0;
        swap(queue, queue_next);
        ++level;
      } // end single
    } // end while
  } // end parallel

  delete [] queue;
  delete [] queue_next;
}

