#include <iostream>

#include <moderngpu/kernel_mergesort.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/transform.hxx>

#include "lca/cuda_commons.h"
#include "lca/cuda_euler_tour.h"
#include "lca/cuda_list_rank.h"

using namespace std;
using namespace mgpu;

void cuda_euler_tour(
    int N, int root, const int *row_offsets, const int *col_indices, int *rank, mgpu::context_t &context )
{
  int *edges_from_to_scan;
  CUCHECK( cudaMalloc( (void **) &edges_from_to_scan, sizeof( int ) * ( N * 2 - 2 ) ) );
  int *edges_from;
  CUCHECK( cudaMalloc( (void **) &edges_from, sizeof( int ) * ( N * 2 - 2 ) ) );

  transform(
      [] MGPU_DEVICE( int thid, int *edges_from_to_scan, const int *row_offsets ) {
        if ( thid != 0 ) edges_from_to_scan[row_offsets[thid]] = 1;
      },
      N,
      context,
      edges_from_to_scan,
      row_offsets );

  scan<scan_type_inc>( edges_from_to_scan, N * 2 - 2, edges_from, context );

  cerr << "Edges from: " << endl;
  CudaPrintTab( edges_from, N * 2 - 2 );

  int *opposite_edges;
  CUCHECK( cudaMalloc( (void **) &opposite_edges, sizeof( int ) * ( N * 2 - 2 ) ) );
  transform( [] MGPU_DEVICE( int thid, int *opposite_edges ) { opposite_edges[thid] = thid; },
             N * 2 - 2,
             context,
             opposite_edges );

  mergesort( opposite_edges,
             N * 2 - 2,
             [=] MGPU_DEVICE( int a, int b ) {
               if ( col_indices[a] != col_indices[b] )
               {
                 return col_indices[a] < col_indices[b];
               }
               return edges_from[a] < edges_from[b];
             },
             context );

  cerr << "Opposite edges: " << endl;
  CudaPrintTab( opposite_edges, N * 2 - 2 );

  int *nextEdge;
  CUCHECK( cudaMalloc( (void **) &nextEdge, sizeof( int ) * ( N * 2 - 2 ) ) );
  transform(
      [=] MGPU_DEVICE( int thid ) {
        int opposite_edge = opposite_edges[thid];
        if ( opposite_edge == N * 2 - 3 || ( edges_from[opposite_edge] != edges_from[opposite_edge + 1] ) )
        {
          nextEdge[thid] = row_offsets[edges_from[opposite_edge]];
        }
        else
        {
          nextEdge[thid] = opposite_edge + 1;
        }
      },
      N * 2 - 2,
      context );

  cerr << "Next: " << endl;
  CudaPrintTab( nextEdge, N * 2 - 2 );

  int listHead;
  CUCHECK( cudaMemcpy( &listHead, row_offsets + root, sizeof( int ), cudaMemcpyDeviceToHost ) );
  cerr << "Head: " << listHead << endl;

  cuda_list_rank( N * 2 - 2, listHead, nextEdge, rank, context );

  context.synchronize();

  cerr << "Rank: " << endl;
  CudaPrintTab( rank, N * 2 - 2 );
}