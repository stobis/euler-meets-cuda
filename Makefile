# Project structure
INCDIR=include
SRCDIR=src
OBJDIR=obj
3RDDIR=3rdparty

# GCC compiler settings
CXX=g++
CXXINC=-I ./$(INCDIR)/
CXXFLAGS=-O2 -std=c++11 -fno-stack-protector $(CXXINC)

# CUDA compiler settings
CUDA=/usr/local/cuda
NVCC=$(CUDA)/bin/nvcc
NVCCSM=sm_50
NVCCINC=-I $(CUDA)/include \
		-I $(CUDA)/samples/common/inc \
		-I ./$(3RDDIR)/cudaweijajalistrank/ \
		-I ./$(3RDDIR)/GpuConnectedComponents/ \
		-I ./$(3RDDIR)/cudabfs/ \
		-I ./$(3RDDIR)/moderngpu/src \
		-I ./$(INCDIR)/ 
		
NVCCFLAGS=-arch $(NVCCSM) -O2 -std=c++11 --expt-extended-lambda -w $(NVCCINC)

LDFLAGS=-L/usr/local/cuda/lib64 -lcudart

OBJFILES=$(patsubst %.cpp, obj/%.o, $(wildcard *.cpp)) \
	$(patsubst %.cpp, obj/%.o, $(wildcard src/**/*.cpp)) \
	$(patsubst %.cu, obj/%.o, $(wildcard src/**/*.cu)) \
	$(patsubst %.cu, obj/%.o, $(wildcard 3rdparty/GpuConnectedComponents/*.cu)) \
	$(patsubst %.cu, obj/%.o, $(wildcard 3rdparty/cudabfs/bfs-mgpu.cu))

# $(info $(OBJFILES))

# Targets
all: runner lca_runner bridges_runner

runner: runner.e
bridges_runner: bridges_runner.e
lca_runner: lca_runner.e

bridges_test:
	cd test/bridges && $(MAKE)

remake: clean all

runner.e: $(OBJFILES) obj/runner.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

bridges_runner.e: obj/bridges_runner.o $(OBJFILES)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

lca_runner.e: $(OBJFILES) obj/lca_runner.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(CXXINC) -c $< -o $@

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(NVCCINC) -c $< -o $@

.PHONY: all clean

clean:
	rm -rf obj *.e
	cd test/bridges && $(MAKE) clean
	cd test/lca && $(MAKE) clean
