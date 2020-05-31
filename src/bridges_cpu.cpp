#include <iostream>
#include <vector>
using namespace std;

#include "timer.hpp"
#include "bridges_cpu.h"

// CPU

void cpu_dfs(const int *row_offsets, const int *col_indices, int v, vector<int> &visited, vector<int> &parent,
             vector<int> &low) {
  static int counter = 0;

  // std::cerr << v << std::endl;

  visited[v] = low[v] = ++counter;

  for (int i = row_offsets[v]; i < row_offsets[v + 1]; i++) {
    int u = col_indices[i];

    // std::cerr << " " << i << " " << u << std::endl;

    if (visited[u] == 0) {
      parent[u] = v;
      cpu_dfs(row_offsets, col_indices, u, visited, parent, low);

      low[v] = min(low[u], low[v]);
    } else if (u != parent[v]) {
      low[v] = min(low[v], visited[u]);
    }

    // std::cerr << " Done" << endl;
  }
}

void cpu_bridges(int N, int M, const int *row_offsets, const int *col_indices, bool *is_bridge) {
  Timer timer("cpu");

  vector<int> visited(N);
  vector<int> parent(N);
  vector<int> low(N);

  cpu_dfs(row_offsets, col_indices, 0, visited, parent, low);

  for (int u = 0; u < N; u++) {
    for (int i = row_offsets[u]; i < row_offsets[u + 1]; i++) {
      int v = col_indices[i];

      int fU = u, fV = v;

      if (parent[v] == u) {
        std::swap(fU, fV);
      }

      if (low[fU] > visited[fV]) {
        is_bridge[i] = true;
      }
    }
  }

  timer.print_and_restart("Overall");
}