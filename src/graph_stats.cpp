#include <iostream>
#include <queue>
#include <vector>

using namespace std;

#include "bridges_cpu.h"
#include "graph.hpp"
#include "timer.hpp"

vector<int> bfs(const Graph &G, int v) {
  vector<int> dist(G.get_N());

  queue<int> Q;
  Q.push(v);

  while (!Q.empty()) {
    int u = Q.front();
    Q.pop();
    for (int i = G.get_row_offsets()[u]; i < G.get_row_offsets()[u + 1]; i++) {
      int a = G.get_col_indices()[i];

      if (dist[a] == 0 && a != v) {
        dist[a] = dist[u] + 1;
        Q.push(a);
      }
    }
  }

  return dist;
}

int get_ecc(const Graph &G, int v) {
  int ecc = 0;
  auto dist = bfs(G, v);

  for (auto d : dist) {
    ecc = max(ecc, d);
  }

  // cerr << "ecc of " << v << " is " << ecc << endl;

  return ecc;
}

bool time_is_gone(Timer &timer, double time_limit_s) {
  return timer.stop_and_get_ms() / 1000.0 > time_limit_s;
}

int get_b(const Graph &G, int i, int u, const vector<int> &dist, Timer &timer, double time_limit_s) {
  int res = 0;
  for (int a = 0; a < dist.size(); a++) {
    if (time_is_gone(timer, time_limit_s))
      return -1;
    if (dist[a] == i) {
      res = max(res, get_ecc(G, a));
    }
  }

  return res;
}

int get_farthest(const Graph &G, int a) {
  auto dist = bfs(G, a);

  int res = 0;
  for (int i = 0; i < dist.size(); i++) {
    if (dist[i] > dist[res])
      res = i;
  }

  return res;
}

int get_node_in_middle(const Graph &G, int a, int b) {
  auto dist = bfs(G, a);
  int res = b;
  while (dist[res] * 2 > dist[b]) {
    for (int i = G.get_row_offsets()[res]; i < G.get_row_offsets()[res + 1]; i++) {
      int v = G.get_col_indices()[i];
      if (dist[v] + 1 == dist[res]) {
        res = v;
        break;
      }
    }
  }

  return res;
}

// iFUB algorithm, returns (lower_bound, upper_bound)
pair<int, int> get_diameter(const Graph &G, double time_limit_s) {
  Timer timer("Diameter");

  int r1 = 0;
  for (int i = 0; i < G.get_N(); i++) {
    if (G.get_row_offsets()[i + 1] - G.get_row_offsets()[i] >
        G.get_row_offsets()[r1 + 1] - G.get_row_offsets()[r1]) {
      r1 = i;
    }
  }

  // r1 = rand() % n;

  int a1 = get_farthest(G, r1);
  int b1 = get_farthest(G, a1);
  int r2 = get_node_in_middle(G, a1, b1);

  int a2 = get_farthest(G, r2);
  int b2 = get_farthest(G, a2);
  int u = get_node_in_middle(G, a2, b2);
  // int u = r1;

  vector<int> dist_u = bfs(G, u);

  int i = get_ecc(G, u);
  int lb = max(max(get_ecc(G, a1), get_ecc(G, a2)), get_ecc(G, u));
  // int lb = get_ecc(G, u);
  int ub = 2 * i;

  while (!time_is_gone(timer, time_limit_s) && ub > lb) {
    int biu = get_b(G, i, u, dist_u, timer, time_limit_s);

    if (biu == -1) // time is gone
      break;

    if (max(lb, biu) > 2 * (i - 1)) {
      int res = max(lb, biu);
      return {res, res};
    } else {
      lb = max(lb, biu);
      ub = 2 * (i - 1);
    }

    i--;
  }
  return {lb, ub};
}

void print_stats(const Graph &G) {
  std::cerr << G.get_N() << ",";
  std::cerr << G.get_M() << ",";

  bool *is_bridge = new bool[G.get_M()];
  cpu_bridges(G.get_N(), G.get_M(), G.get_row_offsets().data(), G.get_col_indices().data(), is_bridge);
  int num_bridges = 0;
  for (int i = 0; i < G.get_M(); i++) {
    if (is_bridge[i])
      num_bridges++;
  }
  std::cerr << num_bridges << ",";
  auto diam = get_diameter(G, 600.0);
  std::cerr << diam.first << ":" << diam.second << std::endl;
}