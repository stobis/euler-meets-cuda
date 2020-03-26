#ifndef GPUTILS_CUH
#define GPUTILS_CUH

#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

#include <moderngpu/context.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/memory.hxx>
using namespace mgpu;

#define dbg 0
#define detailed_time true
#define CUCHECK(x) CudaAssert(x, #x, __FILE__, __LINE__)

typedef long long ll;
typedef unsigned long long ull;

namespace {
template <typename T> void print_device_mem(mem_t<T> &device_mem) {
  // if (!dbg) return;
  cerr << "= print <T>..." << endl;
  vector<T> tmp = from_mem(device_mem);
  for (auto x : tmp) {
    cerr << x << " ";
  }
  cerr << endl;
}

template <> inline void print_device_mem<ll>(mem_t<ll> &device_mem) {
  // if (!dbg) return;
  cerr << "= print edge coded as ll..." << endl;
  vector<ll> tmp = from_mem(device_mem);
  for (auto xd : tmp) {
    ll t = xd;
    int x = (int)t & 0xFFFFFFFF;
    int y = (int)(t >> 32);
    cerr << y << " " << x << endl;
  }
}

inline void get_mem_info() {
  size_t fr, tot;
  cudaMemGetInfo(&fr, &tot);
  cerr << fr / 1e6 << " / " << tot / 1e6 << endl;
}

void _csr_to_list(int N, int M, const int *row_offsets, int *row_indices, mgpu::context_t &context) {
  transform(
      [] MGPU_DEVICE(int index, int const *row_offsets, int *row_indices) {
        if (index == 0)
          return;
        row_indices[row_offsets[index]] = 1;
      },
      N, context, row_offsets, row_indices);

  scan<scan_type_inc>(row_indices, M, row_indices, context);
}

void CudaAssert(cudaError_t error, const char *code, const char *file, int line) {
  if (error != cudaSuccess) {
    cerr << "Cuda error :" << code << ", " << file << ":" << error << endl;
    exit(1);
  }
}

void CudaPrintTab(const int *tab, int size) {
  int *tmp = (int *)malloc(sizeof(int) * size);
  CUCHECK(cudaMemcpy(tmp, tab, sizeof(int) * size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < size; i++) {
    cerr << (tmp[i] < 10 && tmp[i] >= 0 ? " " : "") << tmp[i] << " ";
  }
  cerr << endl;

  free(tmp);
}

void CudaPrintTab(const ull *tab, int size) {
  ull *tmp = (ull *)malloc(sizeof(ull) * size);
  CUCHECK(cudaMemcpy(tmp, tab, sizeof(ull) * size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < size; i++) {
    cerr << (tmp[i] < 10 && tmp[i] >= 0 ? " " : "") << tmp[i] << " ";
  }
  cerr << endl;

  free(tmp);
}
} // namespace

#endif // GPUTILS_CUH
