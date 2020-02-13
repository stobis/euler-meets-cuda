#!/bin/bash

FILES=in/**.bin
EXEC=../../bridges_runner.e

for f in $FILES
do
    echo "=== Test: $f... ==="
    if [ "$1" != "" ]; then
        $EXEC $1 $f
    else
        $EXEC gpu-bfs $f
        $EXEC gpu-tarjan $f
    fi
done
