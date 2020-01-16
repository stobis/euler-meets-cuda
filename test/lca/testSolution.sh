. testVariables.sh

. generateTests.sh

mkdir -p $validityTestsDir

echo "Generating Tests"
for size in ${validityTestsSizes[@]}; do
    genTest $validityTestsDir b $size $size $validityGraspSize 1
done


echo "Generating Answers"

mkdir -p $validityAnswersDir
make $validityOutGenerator.e

for test in $validityTestsDir/*.b.in; do
    testName=$(basename $test)
    outName=$validityAnswersDir/$testName.out
    if [ ! -f $outName ]; then
        echo "Generating $outName"
        ./$validityOutGenerator.e $test $outName 2>/dev/null
    fi
    echo "$outName generated."
done

make $validitySolutionToTest.e

echo "Testing"
for test in $validityTestsDir/*.b.in; do
    testName=$(basename $test)
    outName=$validityAnswersDir/$testName.out

    echo "Testing on $testName"

    # ./$validitySolutionToTest.e $test $validityAnswersDir/out 
    ./$validitySolutionToTest.e $test $validityAnswersDir/out 2>/dev/null

    if diff $validityAnswersDir/out $outName >/dev/null; then
        echo "$testName OK"
    else
        echo "Wrong answer on $testName. Aborting"
        exit 1
    fi
done

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
