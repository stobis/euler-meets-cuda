. testVariables.sh

. generateTests.sh

mkdir -p $validityTestsDir

echo "Generating Tests"
for size in ${validityTestsSizes[@]}; do
    genTest $validityTestsDir b $size $size $validityGraspSize 1
done
echo

echo "Generating Answers"
mkdir -p $validityAnswersDir
for test in $validityTestsDir/*.b.in; do
    testName=$(basename $test)
    outName=$validityAnswersDir/$testName.out
    if [ ! -f $outName ]; then
        echo "Generating $outName"
        $runnerPath -i $test -o $outName -a $validityOutGeneratorAlgorithm 2>/dev/null

        if [ ! -s $outName ]; then
            echo "Error generating answers! $outName is empty!"
            exit 1
        fi
    fi
    echo "$outName generated."
done
echo

for algorithm in ${validitySolutionsToTest[@]}; do
    echo "Testing $algorithm"

    for test in $validityTestsDir/*.b.in; do
        testName=$(basename $test)
        outName=$validityAnswersDir/$testName.out

        echo -n "  Testing on $testName"

        $runnerPath -i $test -o $validityAnswersDir/outTmp.out 2>/dev/null >/dev/null -a $algorithm

        if diff $validityAnswersDir/outTmp.out $outName >/dev/null; then
            echo -e "\t\tOK"
        else
            echo ""
            echo "  Wrong answer on $testName. Aborting"
            exit 1
        fi
    done
done

echo ""
echo "ALL OK"