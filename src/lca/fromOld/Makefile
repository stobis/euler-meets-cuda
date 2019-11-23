NVCC=/usr/local/cuda/bin/nvcc
NVCCFLAGS=-std=c++11 -arch sm_50 -O2 --expt-extended-lambda -I ./3rdparty/moderngpu/src -I ./3rdparty/cudaWeiJaJaListRank

CXX=g++
CXXFLAGS=-std=c++11 -O2 -fno-stack-protector 

all: cudaInlabelLCA.e cudaSimpleLCA.e cpuSimpleLCA.e generateSimple.e generateLongSimple.e cpuRmqLCA.e parseDimacs.e testStats.e parseRoad.e

cpuSimpleLCA.e: cpuSimpleLCA.o commons.o
	$(CXX) $(CXXFLAGS) $^ -o $@

cpuRmqLCA.e: cpuRmqLCA.o commons.o
	$(CXX) $(CXXFLAGS) $^ -o $@

cpuInlabelLCA.e: cpuInlabelLCA.o commons.o
	$(CXX) $(CXXFLAGS) $^ -o $@

cudaInlabelLCA.e: cudaInlabelLCA.cu commons.o cudaCommons.o ./3rdparty/cudaWeiJaJaListRank/cudaWeiJaJaListRank.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

cudaSimpleLCA.e: cudaSimpleLCA.cu commons.o cudaCommons.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

generateSimple.e: generateSimple.o commons.o
	$(CXX) $(CXXFLAGS) $^ -o $@

generateLongSimple.e: generateLongSimple.o commons.o
	$(CXX) $(CXXFLAGS) $^ -o $@

parseDimacs.e: parseDimacs.o commons.o
	$(CXX) $(CXXFLAGS) $^ -o $@

parseRoad.e: parseRoad.o commons.o
	$(CXX) $(CXXFLAGS) $^ -o $@

testStats.e: testStats.o commons.o
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cu %.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: all clean

clean:
	rm -f *.o *.e
