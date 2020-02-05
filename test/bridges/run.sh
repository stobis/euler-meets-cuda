#!/bin/bash

FILES=test/**/*.bin

for f in $FILES
do
    # echo "=> Processing $f... <="
    ./runner.e $f
done
