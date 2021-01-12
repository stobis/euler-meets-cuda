#include <algorithm>
#include <iostream>
#include <vector>
#include "tree.h"

using namespace std;

int main(int argc, char *argv[]) {
  int expectedArgc = 4;
  if (argc < expectedArgc) {
      cerr << "3 args needed: V, Q and Seed. Add filename for binary output" << endl;
      exit(1);
  }

  int V = atoi(argv[1]);
  int Q = atoi(argv[2]);
  int seed = atoi(argv[3]);

  srand(seed + V + Q);

  vector<int> father;
  father.push_back(-1);
  father.push_back(0);

  vector<vector<int>> degreeArray;
  degreeArray.push_back(vector<int>());
  degreeArray.push_back(vector<int>());
  degreeArray[1].push_back(0);
  degreeArray[1].push_back(1);
  int degreesSum = 2;

  while (father.size() < V) {
    int r = rand() % degreesSum;

    int sumDegreeSoFar = 0;
    int fatherDegree = 1;
    while (sumDegreeSoFar + degreeArray[fatherDegree].size() * fatherDegree <= r) {
      sumDegreeSoFar += degreeArray[fatherDegree].size() * fatherDegree;
      fatherDegree++;
    }

    int fatherIndex = (r - sumDegreeSoFar) / fatherDegree;

    int f = degreeArray[fatherDegree][fatherIndex];
    father.push_back(f);

    swap(degreeArray[fatherDegree][fatherIndex],
         degreeArray[fatherDegree][degreeArray[fatherDegree].size() - 1]);
    degreeArray[fatherDegree].pop_back();
    if (degreeArray.size() <= fatherDegree + 1) {
      degreeArray.push_back(vector<int>());
    }
    degreeArray[fatherDegree + 1].push_back(f);
    degreeArray[1].push_back(father.size() - 1);

    degreesSum += 2;
  }
  degreeArray.clear();

  vector<int> shuffledFathers;
  int root;

  shuffleFathers(father, shuffledFathers, root);

  father.clear();

  LcaParentsTree tree(V, root, shuffledFathers);

  shuffledFathers.clear();


  vector<int> q;
  for(int i=0; i<Q*2; i++) {
    q.push_back(rand() % V);
  }
  LcaQueries queries(Q, q);

  q.clear();

  if ( argc == expectedArgc + 1 )
  {
    ofstream out(argv[expectedArgc], ios::binary);
    tree.writeToStream(out);
    queries.writeToStream(out);
  }
  else
  {
    LcaTestCase tc(tree, queries);
    writeToStdOut( tc );
  }
}