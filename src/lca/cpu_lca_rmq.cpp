#include <iostream>
#include <stack>
#include <vector>
#include "lca/commons.h"

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

void cpu_lca_rmq(int N, const int *parents, int Q,
                 const int *queries, int *answers)
{
  Timer timer("Parse Input");

  int root_ = 0;
  for (int i = 0; i < N; i++)
  {
    if (parents[i] == -1)
      root_ = i;
  }
  vector<int> parentsVector(parents, parents + N);
  ParentsTree pt(N, root_, parentsVector);

  vector<int> queriesVector(queries, queries + Q * 2);
  Queries qs(Q, queriesVector);

  TestCase tc(pt, qs);

  timer.measureTime();
  timer.setPrefix("Preprocessing");

  int V = tc.tree.V;
  int root = tc.tree.root;
  INF = V + 10;

  int rmqTabSize = V * 2 - 1;
  int treePower = 1; // for interval tree
  while (treePower < rmqTabSize)
    treePower *= 2;

  rmqTab = new int[treePower * 2];
  preorder = new int[V];
  reversePreorder = new int[V];
  rmqPos = new int[V];

  dfsEulerPath = rmqTab + treePower - 1;
  tree = &tc.tree;
  preCounter = 0;

  vector<int> son(V, -1);
  vector<int> neighbour(V, -1);

  for (int i = 0; i < V; i++)
  {
    if (tc.tree.father[i] == -1)
      continue;
    if (son[tc.tree.father[i]] != -1)
    {
      neighbour[i] = son[tc.tree.father[i]];
    }
    son[tc.tree.father[i]] = i;
  }

  timer.measureTime("Allocs");

  dfsRmq(root, son, neighbour);

  initIntervalTree(rmqTabSize, treePower);

  timer.measureTime("Dfs");
  timer.setPrefix("Queries");

  for (int i = 0; i < tc.q.N; i++)
  {
    int p = rmqPos[preorder[tc.q.tab[i * 2]]];
    int q = rmqPos[preorder[tc.q.tab[i * 2 + 1]]];

    if (p > q)
      swap(p, q);

    answers[i] = reversePreorder[rmqMin(p, q, treePower)];
  }

  timer.measureTime(tc.q.N);
  timer.setPrefix("Write Output");

  timer.measureTime();
}
void dfsRmq(int starting, vector<int> son, vector<int> neighbour)
{
  stack<int> s1;
  s1.push(starting);

  stack<int> s2;

  while (!s1.empty())
  {
    int v = s1.top();

    if (!s2.empty())
    {
      dfsEulerPath[eulerCounter] = preorder[s2.top()];
      rmqPos[preorder[s2.top()]] = eulerCounter;
      eulerCounter++;
    }

    if (!s2.empty() && v == s2.top())
    {
      s1.pop();
      s2.pop();
      continue;
    }

    for (int s = son[v]; s != -1; s = neighbour[s])
    {
      s1.push(s);
    }

    s2.push(v);

    preorder[v] = preCounter;

    reversePreorder[preCounter] = v;

    preCounter++;
  }
}
void initIntervalTree(int n, int power)
{
  for (int i = power + n - 1; i < power * 2 - 1; i++)
  {
    rmqTab[i] = INF;
  }

  for (int i = power - 2; i >= 0; i--)
  {
    rmqTab[i] = min(rmqTab[i * 2 + 1], rmqTab[i * 2 + 2]);
  }
}
int rmqMin(int p, int q, int size)
{
  int *tab = rmqTab - 1;

  p += size;
  q += size;
  int res = tab[p];
  res = min(res, tab[q]);
  while (p / 2 != q / 2)
  {
    if (p % 2 == 0)
      res = min(res, tab[p + 1]);
    if (q % 2 == 1)
      res = min(res, tab[q - 1]);
    p /= 2;
    q /= 2;
  }
  return res;
}