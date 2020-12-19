#include <algorithm>
#include <iostream>
#include <vector>
// #include "tree.h"

using namespace std;

void shuffleFathers(vector<int> &in, vector<int> &out, int &root) {
  int V = in.size();
  vector<int> shuffle;
  for (int i = -2; i < V; i++) {
    shuffle.push_back(i);
  }
  random_shuffle(shuffle.begin(), shuffle.end());

  vector<int> newPos;
  newPos.resize(V);
  for (int i = -2; i < V; i++) {
    newPos[shuffle[i]] = i;
  }

  out.clear();
  for (int i = -2; i < V; i++) {
    if (shuffle[i] == -2) {
      root = i;
    }
    out.push_back(in[shuffle[i]] == -3 ? -1 : newPos[in[shuffle[i]]]);
  }
}

int main(int argc, char *argv[]) {
  int expectedArgc = 4;
  //   if (argc < expectedArgc) {
  //     cerr << "3 args needed: V, Q and Seed. Add filename for binary output" << endl;
  //     exit(1);
  //   }
  int V = atoi(argv[1]);
  //   int Q = atoi(argv[2]);
  //   int seed = atoi(argv[3]);

  //   srand(seed + V + Q);
  srand(V);

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
    while (sumDegreeSoFar + degreeArray[fatherDegree].size() * fatherDegree < r) {
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

  for (int d = 1; d < degreeArray.size(); d++) {
    if (degreeArray[d].size() > 0)
      cout << "Degree = " << d << ", count = " << degreeArray[d].size() << endl;
  }
}