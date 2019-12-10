#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

struct Timer
{
  Timer( string prefix );
  void setPrefix( string prefix );
  void measureTime( string msg = "" );
  void measureTime( int i );

 private:
  clock_t prevTime;
  clock_t prefixTime;
  string prefix;

  double resetTimer( clock_t &timer );
};

struct ParentsTree
{
  int V;
  int root;
  vector<int> father;

  ParentsTree();
  ParentsTree( int V, int root, const vector<int> &father );
  ParentsTree( ifstream &in );

  void writeToStream( ofstream &out );
};

struct Queries
{
  int N;
  vector<int> tab;

  Queries();
  Queries( int N, const vector<int> &tab );
  Queries( ifstream &in );

  void writeToStream( ofstream &out );
};

struct TestCase
{
  ParentsTree tree;
  Queries q;

  TestCase();
  TestCase( const ParentsTree &tree, const Queries &q );
  TestCase( ifstream &in );

  void writeToStream( ofstream &out );
};

int getEdgeCode( int a, bool toFather );  // edge father[a]->a or a->father[a]
int getEdgeStart( ParentsTree &tree, int edgeCode );
int getEdgeEnd( ParentsTree &tree, int edgeCode );

void writeToFile( TestCase &tc, const char *filename );
void writeToStdOut( TestCase &tc );
TestCase readFromFile( const char *filename );
TestCase readFromStdIn();

void writeAnswersToFile( int Q, int *ans, const char *filename );
void writeAnswersToStdOut( int Q, int *ans );

void shuffleFathers( vector<int> &in, vector<int> &out, int &root );

int find( int i, vector<int> &boss );
void junion( int a, int b, vector<int> &boss, vector<int> &father );