#include "lca/commons.h"
#include <iostream>
#include <stack>
#include <vector>

using namespace std;

void dfsRmq(int starting, vector<int> son, vector<int> neighbour);
void initIntervalTree(int n, int power);
int rmqMin(int p, int q, int size);

int INF;

int *rmqTab;
int *dfsEulerPath; // by preorder
int *rmqPos;       // by preorder
int *preorder;
int *reversePreorder;
int preCounter;
int eulerCounter;

ParentsTree *tree;

void cpu_lca_rmq(int N, const int *parents, int Q, const int *queries,
                 int *answers) {
  Timer timer("Parse Input");

  int root = 0;
  for (int i = 0; i < N; i++) {
    if (parents[i] == -1)
      root = i;
  }

  timer.measureTime();
  timer.setPrefix("Preprocessing");

  INF = N + 10;

  int rmqTabSize = N * 2 - 1;
  int treePower = 1; // for interval tree
  while (treePower < rmqTabSize)
    treePower *= 2;

  rmqTab = new int[treePower * 2];
  preorder = new int[N];
  reversePreorder = new int[N];
  rmqPos = new int[N];

  dfsEulerPath = rmqTab + treePower - 1;
  preCounter = 0;

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

  timer.measureTime("Allocs");

  dfsRmq(root, son, neighbour);

  initIntervalTree(rmqTabSize, treePower);

  timer.measureTime("Dfs");
  timer.setPrefix("Queries");

  for (int i = 0; i < Q; i++) {
    int p = rmqPos[preorder[queries[i * 2]]];
    int q = rmqPos[preorder[queries[i * 2 + 1]]];

    if (p > q)
      swap(p, q);

    answers[i] = reversePreorder[rmqMin(p, q, treePower)];
  }

  delete[] rmqTab;
  delete[] preorder;
  delete[] reversePreorder;
  delete[] rmqPos;

  timer.measureTime(Q);
}
void dfsRmq(int starting, vector<int> son, vector<int> neighbour) {
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
void initIntervalTree(int n, int power) {
  for (int i = power + n - 1; i < power * 2 - 1; i++) {
    rmqTab[i] = INF;
  }

  for (int i = power - 2; i >= 0; i--) {
    rmqTab[i] = min(rmqTab[i * 2 + 1], rmqTab[i * 2 + 2]);
  }
}
int rmqMin(int p, int q, int size) {
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