solutionForAnswers=cpuRmqLCA.e
# solutionForAnswers=cpuSimpleLCA.e

make $solutionForAnswers

# testsDir=$(realpath ~/storage/tests)
testsDir=tests

for i in $(ls $testsDir); do
    i=$(basename $i)
    name=$testsDir/${i::-3}.out
    if [[ $i == *.t.in ]]; then
        if [ ! -f $name ]; then
            echo "Generating $name"
            ./$solutionForAnswers <$testsDir/$i >$name
        fi
    fi

    if [[ $i == *.b.in ]]; then
        if [ ! -f $name ]; then
            echo "Generating $name"
            ./$solutionForAnswers $testsDir/$i $name
        fi
    fi
    echo "$name generated."
    echo

done
echo "OK"