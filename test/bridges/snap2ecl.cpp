/*
Copyright (c) 2019 Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted for academic, research, experimental, or personal use provided
that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice,
     this list of conditions, and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions, and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

For all other uses, please contact the Office for Commercialization and Industry
Relations at Texas State University <http://www.txstate.edu/ocir/>.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/


#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <utility>
#include <algorithm>
#include "ECLgraph.h"

int main(int argc, char* argv[])
{
  printf("SNAP to ECL Graph Converter (%s)\n", __FILE__);
  printf("Copyright 2016 Texas State University\n");

  if (argc != 3) {fprintf(stderr, "USAGE: %s input_file_name output_file_name\n\n", argv[0]);  exit(-1);}

  FILE* fin = fopen(argv[1], "rt");  if (fin == NULL) {fprintf(stderr, "ERROR: could not open input file %s\n\n", argv[1]);  exit(-1);}

  char line[256], word1[16], word2[16], word3[16];
  char* ptr = line;
  size_t linesize = 256;
  int cnt;
  do {
    cnt = getline(&ptr, &linesize, fin);
  } while ((cnt > 0) && (strstr(line, "Nodes:") == 0));
  if (cnt <= 0) {fprintf(stderr, "ERROR: could not find line with node and edge counts\n\n");  exit(-1);}
  if ((strstr(line, "Nodes:") == 0) || (strstr(line, "Edges:") == 0)) {fprintf(stderr, "ERROR: could not find line with node and edge counts\n\n");  exit(-1);}

  int nodes = -1, edges = -1;
  cnt = sscanf(line, "%s %s %d %s %d", word1, word2, &nodes, word3, &edges);
  if ((cnt != 5) || (nodes < 1) || (edges < 0) || (strcmp(word1, "#") != 0) || (strcmp(word2, "Nodes:") != 0) || (strcmp(word3, "Edges:") != 0)) {fprintf(stderr, "ERROR: failed to parse nodes and edge counts\n\n");  exit(-1);}

  printf("%s\t#name\n", argv[1]);
  printf("%d\t#nodes\n", nodes);
  printf("%d\t#edges\n", edges);
  printf("no\t#weights\n");

  while (((cnt = getline(&ptr, &linesize, fin)) > 0) && (strstr(line, "#") != 0)) {}
  if (cnt < 3) {fprintf(stderr, "ERROR: could not find non-comment line\n\n");  exit(-1);}

  ECLgraph g;
  g.nodes = nodes;
  g.edges = edges;
  g.nindex = (int*)calloc(nodes + 1, sizeof(int));
  g.eweight = NULL;
  if ((g.nindex == NULL)) {fprintf(stderr, "ERROR: memory allocation failed\n\n");  exit(-1);}

  int src, dst;
  std::vector<std::pair<int, int>> v;
  cnt = 0;
  if (sscanf(line, "%d %d", &src, &dst) == 2) {
    cnt++;
    if ((src < 0) || (src >= nodes)) {fprintf(stderr, "ERROR: source out of range\n\n");  exit(-1);}
    if ((dst < 0) || (dst >= nodes)) {fprintf(stderr, "ERROR: source out of range\n\n");  exit(-1);}
    v.push_back(std::make_pair(src, dst));
    v.push_back(std::make_pair(dst, src));
  }
  while (fscanf(fin, "%d %d", &src, &dst) == 2) {
    cnt++;
    if ((src < 0) || (src >= nodes)) {fprintf(stderr, "ERROR: source out of range\n\n");  exit(-1);}
    if ((dst < 0) || (dst >= nodes)) {fprintf(stderr, "ERROR: source out of range\n\n");  exit(-1);}
    v.push_back(std::make_pair(src, dst));
    v.push_back(std::make_pair(dst, src));
  }
  fclose(fin);
  if (cnt != edges) {fprintf(stderr, "ERROR: failed to read correct number of edges\n\n");  exit(-1);}

  std::sort(v.begin(), v.end());
  // Fix
  auto last = std::unique(v.begin(), v.end());
  v.erase(last, v.end());

  g.edges = v.size();
  g.nlist = (int*)malloc(g.edges * sizeof(int));
  if ((g.nlist == NULL)) {fprintf(stderr, "ERROR: memory allocation failed\n\n");  exit(-1);}
  // Fix end 
  
  // for (int i = 0; i < edges; i++) {
  //   printf("%d %d\n", v[i].first, v[i].second);
  // }

  g.nindex[0] = 0;
  for (int i = 0; i < static_cast<int>(v.size()); i++) {
    int src = v[i].first;
    int dst = v[i].second;
    g.nindex[src + 1] = i + 1;
    g.nlist[i] = dst;
  }

  for (int i = 1; i < (nodes + 1); i++) {
    g.nindex[i] = std::max(g.nindex[i - 1], g.nindex[i]);
  }

  for (int i = 0; i < nodes; i++) {
    if (g.nindex[i + 1] <= g.nindex[i]) {
      printf("err at %d\n", i);
    }
  }

  writeECLgraph(g, argv[2]);
  freeECLgraph(g);

  return 0;
}
