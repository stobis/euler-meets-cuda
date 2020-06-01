#ifndef GRAPH_STATS_HPP
#define GRAPH_STATS_HPP

void print_stats(const Graph &G);
// Calculates diameter of graph. Lower and upper bounds are returned within time limit.
std::pair<int, int> get_diameter(const Graph &G, double time_limit_s);

#endif // GRAPH_STATS_HPP