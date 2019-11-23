#include <iostream>
#include <vector>
#include "commons.h"
using namespace std;

ParentsTree getTreeFromGraph( int V, vector<vector<int> > &G );

int main( int argc, char *argv[] )
{
  ios_base::sync_with_stdio( 0 );
  if ( argc < 2 )
  {
    cerr << "Usage: " << argv[0] << " <IN OUT" << endl;
    exit( 1 );
  }
  int V, E;

  cin >> V >> V >> E;

  vector<vector<int> > G( V );

  for ( int i = 0; i < E; i++ )
  {
    int a, b;
    cin >> a >> b;
    a--;
    b--;
    G[a].push_back( b );
    G[b].push_back( a );
  }

  ParentsTree tree = getTreeFromGraph( V, G );

  srand( 1234 + tree.V );

  int Q = 3000;
  vector<int> q( Q * 2 );
  for ( int i = 0; i < 2 * Q; i++ )
  {
    q[i] = rand() % tree.V;
  }

  Queries queries( Q, q );

  TestCase tc( tree, queries );

  writeToFile( tc, argv[1] );
}
ParentsTree getTreeFromGraph( int V, vector<vector<int> > &G )
{
  vector<int> father( V );
  vector<int> boss( V );
  for ( int i = 0; i < V; i++ )
  {
    boss[i] = i;
  }

  for ( int v = 0; v < V; v++ )
  {
    for ( int i = 0; i < G[v].size(); i++ )
    {
      if ( find( v, boss ) != find( G[v][i], boss ) )
      {
        junion( v, G[v][i], boss, father );
      }
    }
  }

  int numOfConnectedGraphs = 1;
  for ( int i = 1; i < V; i++ )
  {
    if ( find( i - 1, boss ) != find( i, boss ) )
    {
      junion( i - 1, i, boss, father );
      numOfConnectedGraphs++;
    }
  }
  cerr << "V: " << V << ", connected graphs before: " << numOfConnectedGraphs << endl;

  int root = find( 0, boss );
  father[root] = -1;

  return ParentsTree( V, root, father );
}