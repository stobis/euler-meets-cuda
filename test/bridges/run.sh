#!/bin/bash

FILES=in/*.bin

for f in $FILES
do
    echo "=== Test: $f... ==="
    ../../runner_bridges.e $f
done
