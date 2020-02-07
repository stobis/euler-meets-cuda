INCDIR=../../include
# GCC compiler settings
CXX=g++
CXXINC=-I ./$(INCDIR)/
CXXFLAGS=-O2 -std=c++11 -fno-stack-protector $(CXXINC)

all: dimacs2ecl.e mm2ecl.e nr2ecl.e snap2ecl.e shuffle.e

%.e: %.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

.PHONY: all clean

clean:
	rm -rf *.o *.e
