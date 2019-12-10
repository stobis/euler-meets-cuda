#include <cuda_runtime.h>
#include <moderngpu/transform.hxx>

using namespace mgpu;

// Head equals to sum of all-1, root equals to -1. Supports multiple heads,
// trees (calculates depth)
void CudaSimpleListRank(int *devRank, int N, int *devNext, context_t &context);

void CudaAssert(cudaError_t error, const char *code, const char *file,
                int line);

void CudaPrintTab(const int *tab, int size);

#define CUCHECK(x) CudaAssert(x, #x, __FILE__, __LINE__)
