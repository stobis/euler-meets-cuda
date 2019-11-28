#include <moderngpu/transform.hxx>

#include "lca/cuda_commons.h"
#include "lca/cuda_euler_tour.h"

using namespace std;
using namespace mgpu;

void cuda_euler_tour(
    int N, int root, const int *row_offsets, const int *col_indices, int *rank, mgpu::context_t &context )
{
  //TODO
}