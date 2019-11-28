#include <algorithm>
#include <fstream>

#include "lca/commons.h"

using namespace std;

Timer::Timer( string prefix ) : prefix( prefix )
{
  prevTime = clock();
  prefixTime = clock();
}

void Timer::setPrefix( string prefix )
{
  double time = resetTimer( prefixTime );

  cerr.precision( 3 );
  cerr << this->prefix << "," << time << ","
       << "Whole" << endl;

  this->prefix = prefix;
}
void Timer::measureTime( string msg )
{
  double time = resetTimer( prevTime );

  cerr.precision( 3 );
  cerr << prefix << "," << time << "," << msg << endl;
}
void Timer::measureTime( int i )
{
  measureTime( "NumOfQueries:" + to_string( i ) );
}
double Timer::resetTimer( clock_t &timer )
{
  clock_t now = clock();
  double res = double( now - timer ) / CLOCKS_PER_SEC;

  timer = now;

  return res;
}


ParentsTree::ParentsTree() : V( 0 ), root( 0 ) {}
ParentsTree::ParentsTree( int V, int root, const vector<int> &father ) : V( V ), root( root ), father( father ) {}
ParentsTree::ParentsTree( ifstream &in )
{
  in.read( (char *) &V, sizeof( int ) );
  in.read( (char *) &root, sizeof( int ) );

  father.resize( V );
  in.read( (char *) father.data(), sizeof( int ) * V );
}
void ParentsTree::writeToStream( ofstream &out )
{
  out.write( (char *) &V, sizeof( int ) );
  out.write( (char *) &root, sizeof( int ) );
  out.write( (char *) father.data(), sizeof( int ) * V );
}

Queries::Queries() : N( 0 ), tab( vector<int>() ) {}
Queries::Queries( int N, const vector<int> &tab ) : N( N ), tab( tab ) {}
Queries::Queries( ifstream &in )
{
  in.read( (char *) &N, sizeof( int ) );

  tab.resize( N * 2 );
  in.read( (char *) tab.data(), sizeof( int ) * N * 2 );
}
void Queries::writeToStream( ofstream &out )
{
  out.write( (char *) &N, sizeof( int ) );
  out.write( (char *) tab.data(), sizeof( int ) * N * 2 );
}

TestCase::TestCase() : tree( ParentsTree() ), q( Queries() ) {}
TestCase::TestCase( const ParentsTree &tree, const Queries &q ) : tree( tree ), q( q ) {}
TestCase::TestCase( ifstream &in ) : tree( in ), q( in ) {}
void TestCase::writeToStream( ofstream &out )
{
  tree.writeToStream( out );
  q.writeToStream( out );
}

int getEdgeCode( int a, bool toFather )
{
  return a * 2 + toFather;
}
int getEdgeStart( ParentsTree &tree, int edgeCode )
{
  if ( edgeCode % 2 )
    return edgeCode / 2;
  else
    return tree.father[edgeCode / 2];
}
int getEdgeEnd( ParentsTree &tree, int edgeCode )
{
  return getEdgeStart( tree, edgeCode ^ 1 );
}

void writeToFile( TestCase &tc, const char *filename )
{
  ofstream out( filename, ios::binary );
  tc.writeToStream( out );
}
void writeToStdOut( TestCase &tc )
{
  cout << tc.tree.V << " ";
  cout << tc.tree.root << endl;
  for ( int i = 0; i < tc.tree.V; i++ )
  {
    cout << tc.tree.father[i] << " ";
  }
  cout << endl;

  cout << tc.q.N << endl;
  for ( int i = 0; i < tc.q.N * 2; i++ )
  {
    cout << tc.q.tab[i] << " ";
  }
  cout << endl;
}

TestCase readFromFile( const char *filename )
{
  ifstream in( filename, ios::binary );
  return TestCase( in );
}
TestCase readFromStdIn()
{
  ParentsTree tree;
  cin >> tree.V >> tree.root;

  tree.father.resize( tree.V );
  for ( int i = 0; i < tree.V; i++ )
  {
    cin >> tree.father[i];
  }

  int N;
  cin >> N;
  vector<int> q( N * 2 );

  for ( int i = 0; i < N * 2; i++ )
  {
    cin >> q[i];
  }
  return TestCase( tree, Queries( N, q ) );
}

void writeAnswersToStdOut( int Q, int *ans )
{
  for ( int i = 0; i < Q; i++ )
  {
    cout << ans[i] << endl;
  }
}
void writeAnswersToFile( int Q, int *ans, const char *filename )
{
  ofstream out( filename, ios::binary );
  out.write( (char *) ans, sizeof( int ) * Q );
}

void shuffleFathers( vector<int> &in, vector<int> &out, int &root )
{
  int V = in.size();
  vector<int> shuffle;
  for ( int i = 0; i < V; i++ )
  {
    shuffle.push_back( i );
  }
  random_shuffle( shuffle.begin(), shuffle.end() );

  vector<int> newPos;
  newPos.resize( V );
  for ( int i = 0; i < V; i++ )
  {
    newPos[shuffle[i]] = i;
  }

  out.clear();
  for ( int i = 0; i < V; i++ )
  {
    if ( shuffle[i] == 0 )
    {
      root = i;
    }
    out.push_back( in[shuffle[i]] == -1 ? -1 : newPos[in[shuffle[i]]] );
  }
}

int find( int i, vector<int> &boss )
{
  if ( boss[i] != i )
  {
    boss[i] = find( boss[i], boss );
  }
  return boss[i];
}
void junion( int a, int b, vector<int> &boss, vector<int> &father )
{
  father[find( a, boss )] = find( b, boss );
  boss[find( a, boss )] = find( b, boss );
}