#!/bin/bash

FILES=in/**.bin
EXEC=../../bridges_runner.e

if [ "$1" = "-h" ]; then
    echo "Usage: ./run_and_export.sh [naive|tarjan|tarjan-kot|tarjan-bfs|hybrid]"
    exit
fi

for f in $FILES
do
    echo "=== Test: $f... ==="
    if [ "$1" != "" ]; then
        $EXEC $1 $f
    else
        $EXEC naive $f
        $EXEC tarjan $f
        $EXEC tarjan-kot $f
        $EXEC tarjan-bfs $f
        $EXEC hybrid $f
    fi
done
