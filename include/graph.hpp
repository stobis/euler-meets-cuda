#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <fstream>
#include <vector>

typedef std::pair<int, int> edge;

class Graph {
private:
  int n, m;
  std::vector<int> row_offsets, col_indices;

public:
  Graph();
  Graph(std::ifstream &);
  Graph(int, int, std::vector<int>, std::vector<int>);

  int get_N() const;
  int get_M() const;
  std::vector<int> const &get_row_offsets() const;
  std::vector<int> const &get_col_indices() const;

  static Graph read_from_file(const char *);
  static Graph read_from_stdin();
};

#endif // GRAPH_HPP
