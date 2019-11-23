#include <cassert>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <stack>
#include <vector>
#include "commons.h"
using namespace std;

ParentsTree getTreeFromGraph( vector<int> &nodes, vector<int> &edges );
void dfsBuildTree(
    vector<int> &nodes, vector<int> &edges, vector<int> &father, vector<bool> &visited, int a, int depth );

bool printStats = 1;

int main( int argc, char *argv[] )
{
  // From https://github.com/adampolak/cudabfs dimacs-parser.cpp
  if ( argc < 3 )
  {
    cerr << "Usage: " << argv[0] << " IN OUT" << endl;
    exit( 1 );
  }

  vector<int> nodes, edges;

  ifstream in( argv[1] );
  assert( in.is_open() );
  string buf;
  getline( in, buf );
  int n, m;
  sscanf( buf.c_str(), "%d %d", &n, &m );
  nodes.reserve( n + 1 );
  edges.reserve( m );
  nodes.push_back( 0 );
  for ( int i = 0; i < n; ++i )
  {
    getline( in, buf );
    istringstream parser( buf );
    int neighbor;
    while ( parser >> neighbor )
    {
      edges.push_back( neighbor - 1 );
    }
    nodes.push_back( edges.size() );
  }
  assert( edges.size() == m );
  // End From Adam Polak

  ParentsTree tree = getTreeFromGraph( nodes, edges );

  srand( 1234 + tree.V );

  int Q = tree.V / 10;
  vector<int> q( Q * 2 );
  for ( int i = 0; i < 2 * Q; i++ )
  {
    q[i] = rand() % tree.V;
  }

  Queries queries( Q, q );

  TestCase tc( tree, queries );

  writeToFile( tc, argv[2] );
}


ParentsTree getTreeFromGraph( vector<int> &nodes, vector<int> &edges )
{
  int V = nodes.size() - 1;
  vector<int> father( V );
  vector<int> boss( V );
  for ( int i = 0; i < V; i++ )
  {
    boss[i] = i;
  }

  for ( int v = 0; v < V; v++ )
  {
    for ( int i = nodes[v]; i < nodes[v + 1]; i++ )
    {
      if ( find( v, boss ) != find( edges[i], boss ) )
      {
        junion( v, edges[i], boss, father );
      }
    }
  }

  int numOfConnectedGraphs = 0;
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

  for ( int i = 0; i < V; i++ )
  {
    if ( find( i, boss ) != root )
    {
      cerr << "Error processing graph";
      exit( 2 );
    }
  }

  return ParentsTree( V, root, father );
}
