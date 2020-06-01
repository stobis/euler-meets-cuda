#!/bin/bash

FILES=in/**.bin
EXEC=../../bridges_runner.e

if [ "$1" = "" ]; then
    echo "Usage: ./run_graph_stats.sh [output]"
    exit
fi

OUTPUT=($1)
echo "file,N,M,# bridges,diameter_lb:diameter_ub" >$OUTPUT

for f in $FILES
do
  echo "=== Calculating stats for: $f... ==="
  echo -n $f >>$OUTPUT
  echo -n "," >>$OUTPUT
  $EXEC -i $f -s >>/dev/null 2>>$OUTPUT
done