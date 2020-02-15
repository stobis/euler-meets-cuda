#ifndef GPUTILS_CUH
#define GPUTILS_CUH

#include <fstream>
#include <vector>
#include <moderngpu/context.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/memory.hxx>
using namespace mgpu;

#define dbg 0
#define detailed_time true

typedef long long ll;

namespace {
template <typename T>
void print_device_mem(mem_t<T>& device_mem) {
    // if (!dbg) return;
    cerr << "= print <T>..." << endl;
    vector<T> tmp = from_mem(device_mem);
    for (auto x : tmp) {
        cerr << x << " ";
    }
    cerr << endl;
}

template <>
inline void print_device_mem<ll>(mem_t<ll>& device_mem) {
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

void _csr_to_list(int N, int M, const int* row_offsets, int* row_indices,
                  mgpu::context_t& context) {
    transform(
        [] MGPU_DEVICE(int index, int const* row_offsets, int* row_indices) {
            if (index == 0) return;
            row_indices[row_offsets[index]] = 1;
        },
        N, context, row_offsets, row_indices);

    scan<scan_type_inc>(row_indices, M, row_indices, context);
}
}

#endif  // GPUTILS_CUH
