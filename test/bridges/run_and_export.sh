#!/bin/bash

FILES=in/**.bin
EXEC=../../bridges_runner.e

if [ "$1" = "" ] || [ "$2" = "" ]; then
    echo "Usage: ./run_and_export.sh [all|naive|tarjan|hybrid] [output]"
    exit
fi

if [ "$1" != "all" ]; then
    VARIANTS=($1)
    OUTPUT=($2)
else
    VARIANTS=(naive tarjan hybrid cpu multi)
    OUTPUT=($2)
fi

for f in $FILES
do
    echo "=== Test: $f... ==="

    for ix in ${!VARIANTS[*]}
    do
        name="${VARIANTS[$ix]}"
        echo $name
        tmpf="$name.tmp.txt"
        $EXEC -a $name -i $f >> $tmpf
    done
done

for ix in ${!VARIANTS[*]}
do
    name="${VARIANTS[$ix]}"
    tmpf="$name.tmp.txt"
    tmpo="$name.tmp.csv"
    ../stats2csv.py $tmpf $tmpo
    rm $tmpf
done

cat *.tmp.csv > "$2"
rm *.tmp.csv
