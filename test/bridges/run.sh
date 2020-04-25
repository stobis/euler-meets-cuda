#!/bin/bash

FILES=in/**.bin
EXEC=../../bridges_runner.e

if [ "$1" = "-h" ]; then
    echo "Usage: ./run_and_export.sh [naive|tarjan|hybrid]"
    exit
fi

for f in $FILES
do
    echo "=== Test: $f... ==="
    if [ "$1" != "" ]; then
        $EXEC $1 $f
    else
        $EXEC -a naive -i $f
        $EXEC -a tarjan -i $f
        $EXEC -a hybrid -i $f
    fi
done
