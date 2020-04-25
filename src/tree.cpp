#include <algorithm>
#include <fstream>

#include "tree.h"

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

LcaTestCase readFromFile(const char *filename) {
  ifstream in(filename, ios::binary);
  return LcaTestCase(in);
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
void writeAnswersToFile(int Q, int *ans, const char *filename) {
  ofstream out(filename, ios::binary);
  out.write((char *)ans, sizeof(int) * Q);
}