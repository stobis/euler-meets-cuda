#!/bin/bash

FILES=in/**.bin

for f in $FILES
do
    echo "=== Test: $f... ==="
    if [ "$1" != "" ]; then
        ../../runner_bridges.e $1 $f
    else
        ../../runner_bridges.e gpu-bfs $f
        ../../runner_bridges.e gpu-tarjan $f
    fi
done
