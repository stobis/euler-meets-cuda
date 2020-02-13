. testVariables.sh

. generateTests.sh

mkdir -p $validityTestsDir

echo "Generating Tests"
for size in ${validityTestsSizes[@]}; do
    genTest $validityTestsDir $validityTestsEncoding $size $size $validityGraspSize 1
done
echo

echo "Generating Answers"
mkdir -p $validityAnswersDir
for test in $validityTestsDir/*.$validityTestsEncoding.in; do
    testName=$(basename $test)
    outName=$validityAnswersDir/$testName.out
    if [ ! -f $outName ]; then
        echo "Generating $outName"
        $runnerPath -$validityTestsEncoding -i $test -o $outName -a $validityOutGeneratorAlgorithm 2>/dev/null
    fi
    echo "$outName generated."
done
echo

for algorithm in ${validitySolutionsToTest[@]}; do
    echo "Testing $algorithm"

    for test in $validityTestsDir/*.$validityTestsEncoding.in; do
        testName=$(basename $test)
        outName=$validityAnswersDir/$testName.out

        echo -n "  Testing on $testName"

        # ./$validitySolutionToTest.e $test $validityAnswersDir/out 
        # ./$validitySolutionToTest.e $test $validityAnswersDir/out 2>/dev/null
        $runnerPath -$validityTestsEncoding -i $test -o $validityAnswersDir/outTmp.out 2>/dev/null -a $algorithm

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

# # toTest="cudaSimpleLCA"
# toTest="cudaInlabelLCA"
# # toTest="cpuRmqLCA"

# # echo "Generating Tests"
# # ./generateTests.sh 
# echo "Generating Answers"
# ./generateAnswers.sh 

# # testsDir=$(realpath ~/storage/tests)
# testsDir=tests

# make $toTest.e

# for i in $(ls $testsDir/*.b.in); do
#     i=$(basename $i)
#     outName=$testsDir/${i::-3}.out
#     echo "Testing on $i"
#     ./$toTest.e $testsDir/$i out 2>/dev/null
#     if diff out $outName >/dev/null; then
#         echo "$i OK"
#     else
#         echo "Wrong answer on $i. Aborting"
#         exit 1
#     fi
# done

# rm out
# echo "All OK"
