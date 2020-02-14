#!/bin/bash

FILES=in/**.bin
EXEC=../../bridges_runner.e

if [ "$1" = "" ] || [ "$2" = "" ]; then
    echo "Usage: ./run_and_export.sh [naive|tarjan] [output]"
    exit
fi

TMPFILE="$2.tmp.txt"

for f in $FILES
do
    echo "=== Test: $f... ==="
    $EXEC $1 $f >> $TMPFILE
done

./stats2csv.py $TMPFILE "$2"

rm $TMPFILE
