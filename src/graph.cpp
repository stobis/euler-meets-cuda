#include <iostream>

#include "bridges_cpu.h"
#include "graph.hpp"

Graph::Graph() : n(0), m(0) {}

Graph::Graph(std::ifstream &in) {
  in.read(reinterpret_cast<char *>(&n), sizeof(n));
  in.read(reinterpret_cast<char *>(&m), sizeof(m));

  row_offsets.resize(n + 1);
  in.read(reinterpret_cast<char *>(row_offsets.data()), (n + 1) * sizeof(int));
  col_indices.resize(m);
  in.read(reinterpret_cast<char *>(col_indices.data()), (m) * sizeof(int));
}

Graph::Graph(int n, int m, std::vector<int> row_offsets, std::vector<int> col_indices)
    : n(n), m(m), row_offsets(row_offsets), col_indices(col_indices) {}

int Graph::get_N() const { return n; }

int Graph::get_M() const { return m; }

std::vector<int> const &Graph::get_row_offsets() const { return row_offsets; }

std::vector<int> const &Graph::get_col_indices() const { return col_indices; }

Graph Graph::read_from_file(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) {
    std::cerr << "Cannot open file " << filename << std::endl;
    return Graph();
  }
  return Graph(in);
}

Graph Graph::read_from_stdin() {
  int n, m;
  std::vector<int> row_offsets, col_indices;

  std::cin >> n >> m;
  for (int i = 0; i < n + 1; ++i) {
    int a;
    std::cin >> a;
    row_offsets.push_back(a);
  }
  for (int i = 0; i < m; ++i) {
    int b;
    std::cin >> b;
    col_indices.push_back(b);
  }
  return Graph(n, m, row_offsets, col_indices);
}

void Graph::print_stats() const {
  std::cerr << "N: " << n << std::endl;
  std::cerr << "M: " << m << std::endl;

  bool *is_bridge = new bool[m];
  cpu_bridges(n, m, get_row_offsets().data(), get_col_indices().data(), is_bridge);
  int num_bridges = 0;
  for (int i = 0; i < m; i++) {
    if (is_bridge[i])
      num_bridges++;
  }
  std::cerr << "Num bridges: " << num_bridges << std::endl;
}
