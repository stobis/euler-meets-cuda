#!/bin/bash

FILES=in/**.bin
EXEC=../../bridges_runner.e
SINGLE_RUN_SCRIPT=./run_and_export.sh
COMBINE_SCRIPT=../combine_csv.py
TMP_FOLDER=tmpcsv

if [ "$1" = "-h" ]; then
    echo "Usage: ./run_experiments.sh [all|naive|tarjan|hybrid] [times] [output]"
    exit
fi

mkdir $TMP_FOLDER

for ((i=0;i<$2;i++))
do
    $SINGLE_RUN_SCRIPT $1 "$TMP_FOLDER/$i"
done

$COMBINE_SCRIPT $(pwd)/$TMP_FOLDER $3 

rm -rf tmpcsv
