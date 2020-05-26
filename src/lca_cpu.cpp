#include "timer.hpp"
#include "tree.h"
#include <functional>
#include <iostream>
#include <stack>
#include <stdint.h>
#include <string>
#include <vector>
using namespace std;

// CPU LCA INLABEL

void cpu_lca_inlabel(int N, const int *parents, int Q, const int *queries, int *answers) {
  Timer timer("CPU Inlabel");

  int root = 0;
  for (int i = 0; i < N; i++) {
    if (parents[i] == -1)
      root = i;
  }

  vector<int> child(N, -1), neighbour(N, -1);

  for (int i = 0; i < N; ++i) {
    if (parents[i] == -1)
      continue;
    neighbour[i] = child[parents[i]];
    child[parents[i]] = i;
  }

  vector<int> preorder;
  preorder.reserve(N);
  vector<uint32_t> preorder_idx(N), size(N), inlabel(N), ascendant(N), level(N), head(N + 1);

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
    inlabel[u] =
        (preorder_idx[u] + size[u] - 1) &
        (((uint32_t)-1) << (31 - __builtin_clz((preorder_idx[u] - 1) ^ (preorder_idx[u] + size[u] - 1))));
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

  timer.print_and_restart("Preprocessing");

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

  timer.print_and_restart("Queries");
  timer.print_overall();
}

// CPU LCA RMQ

void dfsRmq(int starting, vector<int> son, vector<int> neighbour, int *dfsEulerPath, int *rmqPos,
            int *preorder, int *reversePreorder, int &preCounter, int &eulerCounter);
int rmqMin(int p, int q, int size, int *rmqTab);

void cpu_lca_rmq(int N, const int *parents, int Q, const int *queries, int *answers) {
  Timer timer("CPU RMQ");

  int root = 0;
  for (int i = 0; i < N; i++) {
    if (parents[i] == -1)
      root = i;
  }

  int INF = N + 10;

  int rmqTabSize = N * 2 - 1;
  int treePower = 1; // for interval tree
  while (treePower < rmqTabSize)
    treePower *= 2;

  int *rmqTab = new int[treePower * 2];
  int *preorder = new int[N];
  int *reversePreorder = new int[N];
  int *rmqPos = new int[N];

  int *dfsEulerPath = rmqTab + treePower - 1;
  int preCounter = 0;
  int eulerCounter = 0;

  vector<int> son(N, -1);
  vector<int> neighbour(N, -1);

  for (int i = 0; i < N; i++) {
    if (parents[i] == -1)
      continue;
    if (son[parents[i]] != -1) {
      neighbour[i] = son[parents[i]];
    }
    son[parents[i]] = i;
  }

  dfsRmq(root, son, neighbour, dfsEulerPath, rmqPos, preorder, reversePreorder, preCounter, eulerCounter);

  for (int i = treePower + rmqTabSize - 1; i < treePower * 2 - 1; i++) {
    rmqTab[i] = INF;
  }

  for (int i = treePower - 2; i >= 0; i--) {
    rmqTab[i] = min(rmqTab[i * 2 + 1], rmqTab[i * 2 + 2]);
  }

  timer.print_and_restart("Preprocessing");

  for (int i = 0; i < Q; i++) {
    int p = rmqPos[preorder[queries[i * 2]]];
    int q = rmqPos[preorder[queries[i * 2 + 1]]];

    if (p > q)
      swap(p, q);

    answers[i] = reversePreorder[rmqMin(p, q, treePower, rmqTab)];
  }

  delete[] rmqTab;
  delete[] preorder;
  delete[] reversePreorder;
  delete[] rmqPos;

  timer.print_and_restart("Queries");
  timer.print_overall();
}
void dfsRmq(int starting, vector<int> son, vector<int> neighbour, int *dfsEulerPath, int *rmqPos,
            int *preorder, int *reversePreorder, int &preCounter, int &eulerCounter) {
  stack<int> s1;
  s1.push(starting);

  stack<int> s2;

  while (!s1.empty()) {
    int v = s1.top();

    if (!s2.empty()) {
      dfsEulerPath[eulerCounter] = preorder[s2.top()];
      rmqPos[preorder[s2.top()]] = eulerCounter;
      eulerCounter++;
    }

    if (!s2.empty() && v == s2.top()) {
      s1.pop();
      s2.pop();
      continue;
    }

    for (int s = son[v]; s != -1; s = neighbour[s]) {
      s1.push(s);
    }

    s2.push(v);

    preorder[v] = preCounter;

    reversePreorder[preCounter] = v;

    preCounter++;
  }
}
int rmqMin(int p, int q, int size, int *rmqTab) {
  int *tab = rmqTab - 1;

  p += size;
  q += size;
  int res = tab[p];
  res = min(res, tab[q]);
  while (p / 2 != q / 2) {
    if (p % 2 == 0)
      res = min(res, tab[p + 1]);
    if (q % 2 == 1)
      res = min(res, tab[q - 1]);
    p /= 2;
    q /= 2;
  }
  return res;
}

// CPU LCA SIMPLE

void dfs(int i, vector<int> *G, int *depth);
void cpu_lca_simple(int N, const int *parents, int Q, const int *queries, int *answers) {

  Timer timer("CPU SIMPLE");

  int root = 0;
  for (int i = 0; i < N; i++) {
    if (parents[i] == -1)
      root = i;
  }

  vector<int> *G = new vector<int>[N];
  int *depth = new int[N];
  int *father = new int[N];

  for (int i = 0; i < N; i++) {
    depth[i] = 0;
    int tmp = parents[i];
    father[i] = tmp;
    if (tmp != -1)
      G[tmp].push_back(i);
  }

  depth[root] = 0;
  dfs(root, G, depth);

  timer.print_and_restart("Preprocessing");

  for (int i = 0; i < Q; i++) {
    int p = queries[i * 2];
    int q = queries[i * 2 + 1];
    while (depth[p] != depth[q]) {
      if (depth[p] > depth[q])
        p = father[p];
      else
        q = father[q];
    }

    while (p != q) {
      p = father[p];
      q = father[q];
    }

    answers[i] = p;
  }

  delete[] G;
  delete[] depth;
  delete[] father;

  timer.print_and_restart("Queries");
  timer.print_overall();
}

void dfs(int i, vector<int> *G, int *depth) {
  for (int a = 0; a < G[i].size(); a++) {
    if (depth[G[i][a]] == 0) {
      depth[G[i][a]] = depth[i] + 1;
      dfs(G[i][a], G, depth);
    }
  }
}