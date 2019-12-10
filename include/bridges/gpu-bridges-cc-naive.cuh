#ifndef GPU_BRIDGES_CC_NAIVE_CUH
#define GPU_BRIDGES_CC_NAIVE_CUH

class TestResult;
class Graph;

namespace cc_naive {
TestResult parallel_cc_naive(Graph const&);
}

#endif  // GPU_BRIDGES_CC_CUH
