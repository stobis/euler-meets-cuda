#include <algorithm>
#include <fstream>

#include "commons.h"

using namespace std;

LcaParentsTree::LcaParentsTree() : V(0), root(0) {}
LcaParentsTree::LcaParentsTree(int V, int root, const vector<int> &father)
    : V(V), root(root), father(father) {}
LcaParentsTree::LcaParentsTree(ifstream &in) {
  in.read((char *)&V, sizeof(int));
  in.read((char *)&root, sizeof(int));

  father.resize(V);
  in.read((char *)father.data(), sizeof(int) * V);
}
void LcaParentsTree::writeToStream(ofstream &out) {
  out.write((char *)&V, sizeof(int));
  out.write((char *)&root, sizeof(int));
  out.write((char *)father.data(), sizeof(int) * V);
}

LcaQueries::LcaQueries() : N(0), tab(vector<int>()) {}
LcaQueries::LcaQueries(int N, const vector<int> &tab) : N(N), tab(tab) {}
LcaQueries::LcaQueries(ifstream &in) {
  in.read((char *)&N, sizeof(int));

  tab.resize(N * 2);
  in.read((char *)tab.data(), sizeof(int) * N * 2);
}
void LcaQueries::writeToStream(ofstream &out) {
  out.write((char *)&N, sizeof(int));
  out.write((char *)tab.data(), sizeof(int) * N * 2);
}

LcaTestCase::LcaTestCase() : tree(LcaParentsTree()), q(LcaQueries()) {}
LcaTestCase::LcaTestCase(const LcaParentsTree &tree, const LcaQueries &q) : tree(tree), q(q) {}
LcaTestCase::LcaTestCase(ifstream &in) : tree(in), q(in) {}
void LcaTestCase::writeToStream(ofstream &out) {
  tree.writeToStream(out);
  q.writeToStream(out);
}

void writeToFile(LcaTestCase &tc, const char *filename) {
  ofstream out(filename, ios::binary);
  tc.writeToStream(out);
}
void writeToStdOut(LcaTestCase &tc) {
  cout << tc.tree.V << " ";
  cout << tc.tree.root << endl;
  for (int i = 0; i < tc.tree.V; i++) {
    cout << tc.tree.father[i] << " ";
  }
  cout << endl;

  cout << tc.q.N << endl;
  for (int i = 0; i < tc.q.N * 2; i++) {
    cout << tc.q.tab[i] << " ";
  }
  cout << endl;
}

LcaTestCase readFromFile(bool isBinary, const char *filename) {
  if (isBinary) {
    ifstream in(filename, ios::binary);
    return LcaTestCase(in);
  } else {
    ifstream in(filename);
    return readFromStream(in);
  }
}
LcaTestCase readFromStream(istream &inputStream) {
  LcaParentsTree tree;
  inputStream >> tree.V >> tree.root;

  tree.father.resize(tree.V);
  for (int i = 0; i < tree.V; i++) {
    inputStream >> tree.father[i];
  }

  int N;
  inputStream >> N;
  vector<int> q(N * 2);

  for (int i = 0; i < N * 2; i++) {
    inputStream >> q[i];
  }
  return LcaTestCase(tree, LcaQueries(N, q));
}

LcaTestCase readFromStdIn() { return readFromStream(cin); }

void writeAnswersToStdOut(int Q, int *ans) {
  for (int i = 0; i < Q; i++) {
    cout << ans[i] << endl;
  }
}
void writeAnswersToFile(bool binary, int Q, int *ans, const char *filename) {
  if (binary) {
    ofstream out(filename, ios::binary);
    out.write((char *)ans, sizeof(int) * Q);
  } else {
    ofstream out(filename);
    for (int i = 0; i < Q; i++) {
      out << ans[i] << " ";
    }
    out << endl;
  }
}

void shuffleFathers(vector<int> &in, vector<int> &out, int &root) {
  int V = in.size();
  vector<int> shuffle;
  for (int i = 0; i < V; i++) {
    shuffle.push_back(i);
  }
  random_shuffle(shuffle.begin(), shuffle.end());

  vector<int> newPos;
  newPos.resize(V);
  for (int i = 0; i < V; i++) {
    newPos[shuffle[i]] = i;
  }

  out.clear();
  for (int i = 0; i < V; i++) {
    if (shuffle[i] == 0) {
      root = i;
    }
    out.push_back(in[shuffle[i]] == -1 ? -1 : newPos[in[shuffle[i]]]);
  }
}

int find(int i, vector<int> &boss) {
  if (boss[i] != i) {
    boss[i] = find(boss[i], boss);
  }
  return boss[i];
}
void junion(int a, int b, vector<int> &boss, vector<int> &father) {
  father[find(a, boss)] = find(b, boss);
  boss[find(a, boss)] = find(b, boss);
}