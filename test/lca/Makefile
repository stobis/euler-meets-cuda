# Project structure
SRCDIR="./"
OBJDIR="./"
INCDIR="../../include"


# GCC compiler settings
CXX=g++
CXXINC=-I ./
CXXFLAGS=-O2 -std=c++11 -fno-stack-protector -I $(INCDIR)


OBJFILES=$(patsubst %.cpp, %.o, $(wildcard *.cpp)) \
	$(patsubst %.cpp, %.o, $(wildcard lca/*.cpp)) 

# $(info $(OBJFILES))

# Targets
all: generateSimple.e
remake: clean all
 
generateSimple.e: generateSimple.cpp ../../src/tree.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ 

generateScaleFree.e: generateScaleFree.cpp ../../src/tree.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ 

analyze.e: analyze.cpp ../../src/tree.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ 

.PHONY: all clean

clean:
	rm -rf *.e
	rm -rf *.o
	rm tests -rf
