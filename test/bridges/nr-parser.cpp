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

    ifstream in(argv[1]);
    assert(in.is_open());

    string buf;
    getline(in, buf);
    while (buf[0] == '%') {
        getline(in, buf);
    }

    int n, m;
    sscanf(buf.c_str(), "%d %d %d", &n, &n, &m);

    G.resize(n + 1);
    visited.resize(n + 1);
    label.resize(n + 1);
    cid.resize(n + 1);

    for (int i = 0; i < m; ++i) {
        getline(in, buf);
        istringstream parser(buf);

        int a, b;
        parser >> a >> b;
        if (a == b) continue;
        G[a].push_back(b);
        G[b].push_back(a);
        // in_edges.push_back(make_pair(min(a, b), max(a, b)));
        in_edges.push_back(make_pair(a, b));
    }

    component_id = 1;
    for (int i = 1; i <= n; ++i) {
        // cout << i << endl;
        if (!visited[i]) {
            dfs(i, -1);
            component_id++;
        }
    }
    
    csize.clear();
    csize.resize(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        csize[cid[i]]++;
    }

    auto max_id = max_element(csize.begin(), csize.end()) - csize.begin();

    int id = 1;
    for (int i = 1; i <= n; ++i) {
        if (cid[i] == max_id) {
            label[i] = id++;
        }
    }

    n = id-1;
    m = 0;

    auto & out_edges = in_edges;
    for (auto it = 0; it < in_edges.size(); ++it) {
        auto e = in_edges[it];
        if (cid[e.first] == max_id) {
            out_edges[m++] = make_pair(label[e.first], label[e.second]);
        }
    }

    out_edges.resize(m);

    // cout << n << " " << m << endl;
    // for (auto & e : out_edges) {
    //   cout << e.first << " " << e.second << endl;
    // }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(out_edges.begin(), out_edges.end(), g);

    ofstream out(argv[2], ios::binary);
    assert(out.is_open());

    out.write((char *)&n, sizeof(int));
    out.write((char *)&m, sizeof(int));
    out.write((char *)out_edges.data(), out_edges.size() * sizeof(pii));
}
