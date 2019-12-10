#include <libgen.h>
#include <algorithm>
#include <iostream>
#include <queue>
#include <vector>
#include "commons.h"
using namespace std;

void printCsvTab( vector<pair<int, int> > &tab, string name, string firstRow, string secondRow );

int main( int argc, char *argv[] )
{
  TestCase tc;
  if ( argc == 1 )
  {
    tc = readFromStdIn();
  }
  else
  {
    tc = readFromFile( argv[1] );
  }

  cout << basename( argv[1] ) << endl;

  int V = tc.tree.V;
  int root = tc.tree.root;
  vector<int> &father = tc.tree.father;

  vector<vector<int> > sons( V );

  for ( int i = 0; i < V; i++ )
  {
    if ( father[i] != -1 )
    {
      sons[father[i]].push_back( i );
    }
  }

  vector<int> depth( V );
  depth[root] = 0;

  int numOfLeafs = 0;

  vector<int> leafsOfDepth( V, 0 );
  vector<int> verticesOfDepth( V, 0 );
  int maxLeafDepth = 0;

  queue<int> q;
  q.push( root );

  while ( !q.empty() )
  {
    int v = q.front();
    q.pop();

    if ( sons[v].size() == 0 )
    {
      numOfLeafs++;
      leafsOfDepth[depth[v]]++;

      maxLeafDepth = max( maxLeafDepth, depth[v] );
    }
    verticesOfDepth[depth[v]]++;

    for ( int i = 0; i <sons[v].size(); i++ )
    {
      depth[sons[v][i]] = depth[v] + 1;
      q.push( sons[v][i] );
    }
  }

  int leafsSoFar = 0;
  int verticesSoFar = 0;

  // vector<pair<int, int> > leafsToPrint;
  // vector<pair<int, int> > verticesToPrint;

  // int statsSize = 100;

  // for ( int i = 0; i < maxLeafDepth + 1; i++ )
  // {
  //   leafsSoFar += leafsOfDepth[i];
  //   verticesSoFar += verticesOfDepth[i];

  //   if ( maxLeafDepth < statsSize || i % ( maxLeafDepth / statsSize ) == 0 )
  //   {
  //     leafsToPrint.push_back( make_pair( i, leafsSoFar ) );
  //     verticesToPrint.push_back( make_pair( i, verticesSoFar ) );
  //   }
  // }
  // leafsToPrint.push_back( make_pair( maxLeafDepth-1, leafsSoFar ) );
  // verticesToPrint.push_back( make_pair( maxLeafDepth-1, verticesSoFar ) );



  // printCsvTab( leafsToPrint, "Leafs Depth", "Depth", "Leafs" );

  // cout << endl;

  // printCsvTab( verticesToPrint, "Vetices Depth", "Depth", "Vertices" );

  cout << maxLeafDepth << endl;
}
void printCsvTab( vector<pair<int, int> > &tab, string name, string firstRow, string secondRow )
{
  cout << name << endl;
  cout << firstRow << ",";
  for ( int i = 0; i < tab.size(); i++ )
  {
    cout << tab[i].first << ",";
  }
  cout << endl;

  cout << secondRow << ",";
  for ( int i = 0; i < tab.size(); i++ )
  {
    cout << tab[i].second << ",";
  }
  cout << endl;
}