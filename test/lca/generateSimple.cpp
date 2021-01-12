#include <algorithm>
#include <iostream>
#include "tree.h"

using namespace std;

int main( int argc, char* argv[] )
{
  int expectedArgc = 5;
  if ( argc < expectedArgc )
  {
    cerr << "4 args needed: V, Q, graspSize and Seed. Add filename for binary output" << endl;
    exit( 1 );
  }
  int V = atoi( argv[1] );
  int Q = atoi( argv[2] );
  int graspSize = atoi( argv[3] );
  int seed = atoi( argv[4] );

  srand( seed + V + Q );

  vector<int> tab;
  tab.push_back( -1 );
  for ( int i = 1; i < V; i++ )
  {
    int minFather = graspSize == -1 ? -1 : i - graspSize;
    if ( minFather < 0 ) minFather = 0;

    tab.push_back( ( rand() % ( i - minFather ) ) + minFather );
  }

  vector<int> father;
  int root;

  shuffleFathers( tab, father, root );

  LcaParentsTree tree( V, root, father );

  vector<int> q;
  for ( int i = 0; i < Q * 2; i++ )
  {
    q.push_back( rand() % V );
  }
  LcaQueries queries( Q, q );

  LcaTestCase tc( tree, queries );
  if ( argc == expectedArgc + 1 )
    writeToFile( tc, argv[expectedArgc] );
  else
    writeToStdOut( tc );
}