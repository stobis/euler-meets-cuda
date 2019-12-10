#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <fstream>
#include <vector>

typedef std::pair<int, int> edge;

class Graph {
   private:
    int n, m;
    std::vector<std::pair<int, int>> edges;

   public:
    Graph(std::ifstream&);
    Graph(int, int, std::vector<std::pair<int, int>>);

    int get_N() const;
    int get_M() const;
    // int get_M();
    std::vector<std::pair<int, int>> const& get_Edges() const;

    static Graph read_from_file(const char*);
    static Graph read_from_stdin();
};

#endif  // GRAPH_HPP
