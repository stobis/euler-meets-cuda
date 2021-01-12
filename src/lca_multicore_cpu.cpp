#include "timer.hpp"

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <memory>
#include <vector>

#include <parallel/algorithm>
#include <parallel/numeric>

using namespace std;

namespace emc {

// MULTICORE CPU LCA INLABEL
void multicore_cpu_lca_inlabel(int N, const int *parents, int Q, const int *queries, int *answers, int batch_size) {
  Timer timer("Multicore CPU Inlabel");

  int root = 0;
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    if (parents[i] == -1)
      root = i;
  }
  unique_ptr<int[]> edges_from(new int[2 * (N - 1)]);
  unique_ptr<int[]> edges_to(  new int[2 * (N - 1)]);
  #pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    if (i == root)
      continue;
    int afterRoot = i > root;
    edges_from[2 * (i - afterRoot) + 0] = parents[i];
    edges_to[  2 * (i - afterRoot) + 0] = i;
    edges_from[2 * (i - afterRoot) + 1] = i;
    edges_to[  2 * (i - afterRoot) + 1] = parents[i];
  }

  unique_ptr<int[]> edges_sorted(new int[2 * (N - 1)]);
  #pragma omp parallel for
  for (int i = 0; i < 2 * (N - 1); ++i) {
    edges_sorted[i] = i;
  }
  __gnu_parallel::sort(edges_sorted.get(), edges_sorted.get() + 2 * (N - 1),
      [&](int a, int b){ return edges_from[a] < edges_from[b];});

  unique_ptr<int[]> first(new int[N]);
  #pragma omp parallel for
  for (int i = 0; i < 2 * (N - 1); ++i) {
    if (i == 0 || edges_from[edges_sorted[i - 1]] != edges_from[edges_sorted[i]])
      first[edges_from[edges_sorted[i]]] = edges_sorted[i];
  }

  unique_ptr<int[]> next(new int[2 * (N - 1)]);
  #pragma omp parallel for
  for (int i = 0; i < 2 * (N - 1); ++i) {
    if (i == 2 * (N - 1) - 1 || edges_from[edges_sorted[i + 1]] != edges_from[edges_sorted[i]])
      next[edges_sorted[i]] = first[edges_from[edges_sorted[i]]];
    else
      next[edges_sorted[i]] = edges_sorted[i + 1];
  }

  unique_ptr<int[]> succ(new int[2 * (N - 1)]);
  #pragma omp parallel for
  for (int i = 0; i < 2 * (N - 1); ++i) {
    assert(edges_from[i] == edges_from[next[i]]);
    int twin = i ^ 1;
    succ[i] = next[twin];
  }

  typedef array<int, 2> EulerTourItem;
  auto add = [](EulerTourItem a, EulerTourItem b) { return EulerTourItem{a[0] + b[0], a[1] + b[1]}; };

  unique_ptr<EulerTourItem[]> pref(new EulerTourItem[2 * (N - 1)]);
  #pragma omp parallel for
  for (int i = 0; i < 2 * (N - 1); ++i) {
    pref[i][0] = (i % 2 == 0);  // is edge going down?
    pref[i][1] = 1 - pref[i][0];
  }
  
  int n_checkpoints = min(100, 2 * (N - 1));
  vector<int> checkpoints(n_checkpoints);
  vector<bool> is_checkpoint(2 * (N - 1));
  checkpoints[0] = first[root];
  is_checkpoint[first[root]] = true;
  for (int i = 1; i < n_checkpoints; ++i) {
    int& checkpoint = checkpoints[i];
    do {
      checkpoint = rand() % (2 * (N - 1));
    } while (is_checkpoint[checkpoint]);
    is_checkpoint[checkpoint] = true;
  }

  vector<int> checkpoint_succ(n_checkpoints);
  vector<EulerTourItem> offset(n_checkpoints);

  #pragma omp parallel for
  for (int i = 0; i < n_checkpoints; ++i) {
    int u = checkpoints[i];
    EulerTourItem total = {0, 0};
    do {
      pref[u] = add(total, pref[u]);
      swap(total, pref[u]);
      u = succ[u];
    } while (!is_checkpoint[u]);
    checkpoint_succ[i] = find(checkpoints.begin(), checkpoints.end(), u) - checkpoints.begin();
    offset[checkpoint_succ[i]] = total;
  }

  offset[0] = {0, 0};
  for (int i = 0; checkpoint_succ[i] != 0; i = checkpoint_succ[i])
    offset[checkpoint_succ[i]] = add(offset[checkpoint_succ[i]], offset[i]);
  
  #pragma omp parallel for
  for (int i = 0; i < n_checkpoints; ++i) {
    int u = checkpoints[i];
    do {
      pref[u] = add(pref[u], offset[i]);
      u = succ[u];
    } while (!is_checkpoint[u]);
  }

  unique_ptr<uint32_t[]> preorder_idx(new uint32_t[N]);
  unique_ptr<uint32_t[]> size(new uint32_t[N]);
  unique_ptr<uint32_t[]> level(new uint32_t[N]);
  unique_ptr<uint32_t[]> inlabel(new uint32_t[N]);

  #pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    if (i == root) {
      preorder_idx[i] = 0;
      size[i] = N;
      level[i] = 0;
    } else {
      int first_edge = 2 * (i - (i > root));
      int last_edge = first_edge + 1;
      preorder_idx[i] = 1 + pref[first_edge][0];
      size[i] = pref[last_edge][0] - pref[first_edge][0];
      level[i] = pref[last_edge][0] - pref[last_edge][1];
    }
    inlabel[i] =
        (preorder_idx[i] + size[i]) &
        (((uint32_t)-1) << (31 - __builtin_clz((preorder_idx[i]) ^ (preorder_idx[i] + size[i]))));
  }

  unique_ptr<uint32_t[]> head(new uint32_t[N + 1]);

  #pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    if (i == root || inlabel[i] != inlabel[parents[i]])
      head[inlabel[i]] = i;
  }

  unique_ptr<int[]> pref_ascendant(new int[2 * (N - 1)]);
  memset(pref_ascendant.get(), 0, sizeof(int) * 2 * (N - 1));  // faster than setting via openmp
  for (int i = 0; i < 2 * (N - 1); ++i) {
    int j = pref[i][0] + pref[i][1];
    if (inlabel[edges_from[i]] != inlabel[edges_to[i]]) {
      if (i % 2 == 0)
        pref_ascendant[j] = 1 << __builtin_ctz(inlabel[edges_to[i]]);
      else
        pref_ascendant[j] = -(1 << __builtin_ctz(inlabel[edges_from[i]]));
    }
  }
  pref_ascendant[0] += (1 << __builtin_ctz(inlabel[root]));
  __gnu_parallel::partial_sum(pref_ascendant.get(), pref_ascendant.get() + 2 * (N - 1), pref_ascendant.get());

  unique_ptr<uint32_t[]> ascendant(new uint32_t[N]);
  #pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    if (i == root) {
      ascendant[i] = 1 << __builtin_ctz(inlabel[root]);
    } else {
      int first_edge = 2 * (i - (i > root));
      int first_edge_et_idx = pref[first_edge][0] + pref[first_edge][1];
      ascendant[i] = pref_ascendant[first_edge_et_idx];
    }
  }

  timer.print_and_restart("Preprocessing");

  for (int batch_start = 0; batch_start < Q; batch_start += batch_size) {
    int batch_end = min(Q, batch_start + batch_size);
    #pragma omp parallel for
    for (int i = batch_start; i < batch_end; i++) {
      int x = queries[2 * i];
      int y = queries[2 * i + 1];
      if (inlabel[x] != inlabel[y]) {
        int i = max(31 - __builtin_clz(inlabel[x] ^ inlabel[y]),
                    max(__builtin_ctz(inlabel[x]), __builtin_ctz(inlabel[y])));
        uint32_t b = inlabel[x] & (((uint32_t)-1) << i);
        uint32_t common = ((ascendant[x] & ascendant[y]) >> i) << i;
        int j = __builtin_ctz(common);
        uint32_t inlabel_z = ((inlabel[x] >> j) | 1) << j;
        auto climb = [&](int u) {
          if (inlabel[u] == inlabel_z)
            return u;
          int k = 31 - __builtin_clz(ascendant[u] & ((1 << j) - 1));
          uint32_t inlabel_w = ((inlabel[u] >> k) | 1) << k;
          return parents[head[inlabel_w]];
        };
        x = climb(x);
        y = climb(y);
      }
      answers[i] = level[x] <= level[y] ? x : y;
    }
  }

  timer.print_and_restart("Queries");
  timer.print_overall();
}

} // namespace emc
