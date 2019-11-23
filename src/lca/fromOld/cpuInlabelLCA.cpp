#include "commons.h"

#include <stdint.h>

#include <functional>
#include <iostream>
#include <vector>
using namespace std;

int main(int argc, char *argv[]) {
  Timer timer("Parse Input");

  TestCase tc = argc == 1 ? readFromStdIn() : readFromFile(argv[1]);

  timer.measureTime();
  timer.setPrefix("Preprocessing");

  int V = tc.tree.V;

  vector<int> child(V, -1), neighbour(V, -1);
  
  for (int i = 0; i < V; ++i) {
    if (tc.tree.father[i] == -1)
      continue;
    neighbour[i] = child[tc.tree.father[i]];
    child[tc.tree.father[i]] = i;
  }

  vector<int> preorder;
  preorder.reserve(V);
  vector<uint32_t> preorder_idx(V), size(V), inlabel(V), ascendant(V), level(V), head(V + 1);

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
    inlabel[u] = (preorder_idx[u] + size[u] - 1) & (((uint32_t)-1) << (
      31 - __builtin_clz((preorder_idx[u] - 1 ) ^ (preorder_idx[u] + size[u] - 1))));
  };

  level[tc.tree.root] = 0;
  dfs(tc.tree.root);

  for (int u : preorder) {
    int p = tc.tree.father[u];
    ascendant[u] = p == -1 ? 0 : ascendant[p];
    if (p == -1 || inlabel[u] != inlabel[p]) {
      ascendant[u] += 1 << __builtin_ctz(inlabel[u]);
      head[inlabel[u]] = u;
    }
  }

  timer.measureTime();
  timer.setPrefix("Queries");

  vector<int> answers(tc.q.N);

  for (int i = 0; i < tc.q.N; i++)  {
    int x = tc.q.tab[2 * i];
    int y = tc.q.tab[2 * i + 1];
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
        return tc.tree.father[head[inlabel_w]];
      };
      x = climb(x);
      y = climb(y);
    }
    answers[i] = level[x] <= level[y] ? x : y;
  }

  timer.measureTime(tc.q.N);
  timer.setPrefix("Write Output");

  if (argc < 3)
    writeAnswersToStdOut(tc.q.N, answers.data());
  else
    writeAnswersToFile(tc.q.N, answers.data(), argv[2]);

  timer.measureTime();
}
