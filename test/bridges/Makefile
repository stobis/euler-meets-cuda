ROOTDIR=./../..
INCDIR=${ROOTDIR}/include
SRCDIR=${ROOTDIR}/src
# GCC compiler settings
CXX=g++
CXXINC=-I ./$(INCDIR)/
CXXFLAGS=-O2 -std=c++11 -fno-stack-protector $(CXXINC)

all: dimacs2ecl.e mm2ecl.e snap2ecl.e connect.e sequential.e

%.e: %.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

sequential.e: sequential.cpp ${SRCDIR}/graph.cpp ${SRCDIR}/timer.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

.PHONY: all clean

clean:
	rm -rf *.o *.e
