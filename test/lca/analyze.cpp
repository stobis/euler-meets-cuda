#include <iostream>
#include <vector>
#include "tree.h"
using namespace std;

vector<int> depths;
vector<vector<int>>G;
vector<int> degrees;

void dfs(int a, int depth) {
    if(depths.size() <= depth) {
        depths.push_back(0);
    }

    depths[depth]++;
    for(int i=0; i<G[a].size(); i++) {
        dfs(G[a][i], depth+1);
    }
}

int main(int argc, char *argv[]) {
    LcaTestCase tc = readFromFile(argv[1]);

    G.resize(tc.tree.V);

    for(int i=0; i<tc.tree.V; i++) {
        int father = tc.tree.father[i];
        if(father != -1) {
            G[father].push_back(i);
        }
    }

    tc.tree.father.clear();

    // for(int i=0; i<G.size(); i++) {
    //     while(degrees.size() <= G[i].size()) {
    //         degrees.push_back(0);
    //     }
    //     degrees[G[i].size()]++;
    // }

    // for(int i=0; i<degrees.size(); i++) {
    //     if(degrees[i] > 0) {
    //         cout<<i+1<<", "<<degrees[i]<<endl;
    //     }
    // }

    long long d = 0;

    cout<<G.size()<<" A"<<endl;
    dfs(tc.tree.root, 0);
    for(int i=0; i<depths.size(); i++) {
        d+=depths[i] * (i+1);
    }

    cout<< ((long double) d) / tc.tree.V<<endl;
}