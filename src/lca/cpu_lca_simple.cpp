#include "lca/commons.h"
#include <iostream>
#include <vector>

using namespace std;

void dfs(int i);

vector<int> *G;
int *depth;
int *father;
int *queries;

void cpu_lca_simple(int N, const int *parents, int Q, const int *queries,
                    int *answers) {
  Timer timer("Parse Input");

  int root = 0;
  for (int i = 0; i < N; i++) {
    if (parents[i] == -1)
      root = i;
  }

  G = new vector<int>[N];
  depth = new int[N];
  father = new int[N];

  for (int i = 0; i < N; i++) {
    depth[i] = 0;
    int tmp = parents[i];
    father[i] = tmp;
    if (tmp != -1)
      G[tmp].push_back(i);
  }

  depth[root] = 0;
  dfs(root);

  timer.measureTime();
  timer.setPrefix("Queries");

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

  timer.measureTime(Q);
}

void dfs(int i) {
  for (int a = 0; a < G[i].size(); a++) {
    if (depth[G[i][a]] == 0) {
      depth[G[i][a]] = depth[i] + 1;
      dfs(G[i][a]);
    }
  }
}
