#include "lca/commons.h"

#include <stdint.h>

#include <functional>
#include <iostream>
#include <vector>
using namespace std;

void cpu_lca_inlabel(int N, const int *parents, int Q, const int *queries,
                     int *answers) {
  Timer timer("Parse Input");

  int root = 0;
  for (int i = 0; i < N; i++) {
    if (parents[i] == -1)
      root = i;
  }

  timer.measureTime();
  timer.setPrefix("Preprocessing");

  vector<int> child(N, -1), neighbour(N, -1);

  for (int i = 0; i < N; ++i) {
    if (parents[i] == -1)
      continue;
    neighbour[i] = child[parents[i]];
    child[parents[i]] = i;
  }

  vector<int> preorder;
  preorder.reserve(N);
  vector<uint32_t> preorder_idx(N), size(N), inlabel(N), ascendant(N), level(N),
      head(N + 1);

  uint32_t dfstime = 1;

  function<void(int)> dfs = [&](int u) {
    preorder.push_back(u);
    preorder_idx[u] = dfstime++;
    size[u] = 1;
    for (int v = child[u]; v != -1; v = neighbour[v]) {
      level[v] = level[u] + 1;
      dfs(v);
      size[u] += size[v];
    }
    inlabel[u] = (preorder_idx[u] + size[u] - 1) &
                 (((uint32_t)-1)
                  << (31 - __builtin_clz((preorder_idx[u] - 1) ^
                                         (preorder_idx[u] + size[u] - 1))));
  };

  level[root] = 0;
  dfs(root);

  for (int u : preorder) {
    int p = parents[u];
    ascendant[u] = p == -1 ? 0 : ascendant[p];
    if (p == -1 || inlabel[u] != inlabel[p]) {
      ascendant[u] += 1 << __builtin_ctz(inlabel[u]);
      head[inlabel[u]] = u;
    }
  }

  timer.measureTime();
  timer.setPrefix("Queries");

  for (int i = 0; i < Q; i++) {
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

  timer.measureTime(Q);
}
