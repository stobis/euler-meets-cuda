#include <iostream>
#include <vector>
#include "lca/commons.h"

using namespace std;

void dfs(int i);

vector<int> *G;
int *depth;
int *father;
int *queries;

void cpu_lca_simple(int N, const int *parents, int Q,
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

  timer.measureTime("Read");

  int V = tc.tree.V;
  int root = tc.tree.root;

  G = new vector<int>[V];
  depth = new int[V];
  father = new int[V];

  for (int i = 0; i < V; i++)
  {
    depth[i] = 0;
    int tmp = tc.tree.father[i];
    father[i] = tmp;
    if (tmp != -1)
      G[tmp].push_back(i);
  }

  timer.measureTime("Parse");
  timer.setPrefix("Preprocessing");

  depth[root] = 0;
  dfs(root);

  timer.measureTime();
  timer.setPrefix("Queries");

  // for ( int i = 0; i < V; i++ )
  // {
  //   cout << i << ": " << depth[i] << endl;
  // }

  queries = tc.q.tab.data();

  for (int i = 0; i < Q; i++)
  {
    int p = queries[i * 2];
    int q = queries[i * 2 + 1];
    while (depth[p] != depth[q])
    {
      if (depth[p] > depth[q])
        p = father[p];
      else
        q = father[q];
    }

    while (p != q)
    {
      p = father[p];
      q = father[q];
    }

    answers[i] = p;
  }

  timer.measureTime(Q);
  timer.setPrefix("Write Output");

  timer.measureTime();
}

void dfs(int i)
{
  for (int a = 0; a < G[i].size(); a++)
  {
    if (depth[G[i][a]] == 0)
    {
      depth[G[i][a]] = depth[i] + 1;
      dfs(G[i][a]);
    }
  }
}
