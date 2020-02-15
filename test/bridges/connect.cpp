#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <random>
using namespace std;

typedef pair<int, int> pii;
std::vector<std::vector<int>> G;
std::vector<bool> visited;
std::vector<int> label;
std::vector<int> cid;
std::vector<int> csize;
int component_id;

void dfs(int start, int parent) {
    visited[start] = true;
    cid[start] = component_id;

    for (auto x : G[start]) {
        if (x == parent) continue;
        if (!visited[x]) {
            dfs(x, start);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " IN OUT" << endl;
        exit(1);
    }

    vector<pii> in_edges;

    ifstream in(argv[1], ios::binary);
    assert(in.is_open());

    int n, m;
    in.read((char *)(&n), sizeof(n));
    in.read((char *)(&m), sizeof(m));

    int const oldN = n;
    int const oldM = m;

    vector<int> row_offsets(n+1);
    vector<int> col_indices(m);

    in.read((char *)(row_offsets.data()), (n+1)*sizeof(int));
    in.read((char *)(col_indices.data()), (m)*sizeof(int));

    G.resize(n + 1);
    visited.resize(n + 1);
    label.resize(n + 1);
    cid.resize(n + 1);

    for (int i = 0; i < n; ++i) {
        
        for (int j = row_offsets[i]; j < row_offsets[i+1]; ++j) {
            int a = i, b = col_indices[j];
            if (a == b) continue;
            G[a].push_back(b);
            G[b].push_back(a);
            in_edges.push_back(make_pair(a, b)); 
        }
    }
    row_offsets.clear();
    col_indices.clear();

    component_id = 1;
    for (int i = 0; i < n; ++i) {
        // cout << i << endl;
        if (!visited[i]) {
            dfs(i, -1);
            component_id++;
        }
    }
    
    csize.clear();
    csize.resize(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        csize[cid[i]]++;
    }

    auto max_id = max_element(csize.begin(), csize.end()) - csize.begin();

    int id = 0;
    for (int i = 0; i < n; ++i) {
        if (cid[i] == max_id) {
            label[i] = id++;
        }
    }

    n = id;
    m = 0;

    auto & out_edges = in_edges;
    for (auto it = 0; it < in_edges.size(); ++it) {
        auto e = in_edges[it];
        if (cid[e.first] == max_id) {
            out_edges[m++] = make_pair(label[e.first], label[e.second]);
        }
    }

    out_edges.resize(m);
    std::sort(out_edges.begin(), out_edges.end());

    row_offsets.resize(n+1);
    col_indices.resize(m);

    row_offsets[0] = 0;
    for (int i = 0; i < m; i++) {
        int src = out_edges[i].first;
        int dst = out_edges[i].second;
        row_offsets[src + 1] = i + 1;
        col_indices[i] = dst;
    }

    for (int i = 1; i < (n + 1); i++) {
        row_offsets[i] = std::max(row_offsets[i - 1], row_offsets[i]);
    }

    // cout << n << " " << m << endl;
    // for (auto & e : out_edges) {
    //   cout << e.first << " " << e.second << endl;
    // }

    // std::random_device rd;
    // std::mt19937 g(rd());
    // std::shuffle(out_edges.begin(), out_edges.end(), g);

    ofstream out(argv[2], ios::binary);
    assert(out.is_open());

    out.write((char *)&n, sizeof(int));
    out.write((char *)&m, sizeof(int));
    out.write((char *)row_offsets.data(), row_offsets.size()*sizeof(int));
    out.write((char *)col_indices.data(), col_indices.size()*sizeof(int));
    out.close();
    std::cout << "#components: " << component_id - 1 << std::endl;
    std::cout << "#old (N, M): " << oldN << " " << oldM << std::endl;
    std::cout << "#new (N, M): " << n << " " << m << std::endl;
}
