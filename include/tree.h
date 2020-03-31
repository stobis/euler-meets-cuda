#ifndef EMC_TREE_HPP_
#define EMC_TREE_HPP_

#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

struct LcaParentsTree {
  int V;
  int root;
  vector<int> father;

  LcaParentsTree();
  LcaParentsTree(int V, int root, const vector<int> &father);
  LcaParentsTree(ifstream &in);

  void writeToStream(ofstream &out);
};

struct LcaQueries {
  int N;
  vector<int> tab;

  LcaQueries();
  LcaQueries(int N, const vector<int> &tab);
  LcaQueries(ifstream &in);

  void writeToStream(ofstream &out);
};

struct LcaTestCase {
  LcaParentsTree tree;
  LcaQueries q;

  LcaTestCase();
  LcaTestCase(const LcaParentsTree &tree, const LcaQueries &q);
  LcaTestCase(ifstream &in);

  void writeToStream(ofstream &out);
};

void writeToFile(LcaTestCase &tc, const char *filename);
void writeToStdOut(LcaTestCase &tc);
LcaTestCase readFromFile(const char *filename);
LcaTestCase readFromStream(istream &inputStream);
LcaTestCase readFromStdIn(); 

void writeAnswersToFile(int Q, int *ans, const char *filename);
void writeAnswersToStdOut(int Q, int *ans);

#endif // EMC_TREE_HPP_