. testVariables.sh 

. generateTests.sh

for T in ${testsToRun[@]}; do
    echo "Running Experiment $T"

    echo "Generating Tests"
    _testsDir=E${T}TestsDir
    mkdir -p ${!_testsDir}

    eval _testSizes=( '"${E'${T}'TestSizes[@]}"' )
    eval _graspSizes=( '"${E'${T}'GraspSizes[@]}"' )
    _numOfSeeds=E${T}DifferentSeeds
    for size in ${_testSizes[@]}; do
        for graspSize in ${_graspSizes}; do
            genTest ${!_testsDir} $timedTestsEncoding $size $size $graspSize ${!_numOfSeeds}
        done
    done

    _resultsDir=E${T}ResultsDir
    mkdir -p ${!_resultsDir}

    eval _solutionsToTest=( '"${E'${T}'SolutionsToTest[@]}"' )
    for solution in ${_solutionsToTest[@]}; do
        echo "Testing $solution"
        for test in ${!_testsDir}/*.$timedTestsEncoding.in; do
            testName=$(basename $test)
            testOutName=$(echo $testName | sed 's/s[0-9][0-9]*.//g') 
            outName=${!_resultsDir}/$solution\$${testOutName::-5}\$$(date '+%Y-%m-%d-%H-%M-%S' -d @$(stat -c %Y $runnerPath)).out

            echo -e "\tRunning $(basename $outName)"

            touch $outName
            timeout $singleRunTimeout $runnerPath -$timedTestsEncoding -i $test -o /dev/null -a $solution 2>>$outName
            echo "" >>$outName
        done
    done

done

# exit

# if $runE1; then
#     echo "Running Experiment 1"

#     if $E1PurgeExistingTests; then
#         echo "Purging Existing Tests"
#         rm $E1TestsDir -rf
#     fi

#     echo "Generating Tests"
#     mkdir -p $E1TestsDir

#     for size in ${E1TestSizes[@]}; do
#         for graspSize in ${E1GraspSizes[@]}; do
#             genTest $E1TestsDir b $size $(($size)) $graspSize $E1DifferentSeeds
#         done
#     done

#     mkdir -p $E1ResultsDir

#     for solution in ${E1SolutionsToTest[@]}; do
#         make $solution.e
#         echo "Testing $solution"
#         for test in $E1TestsDir/*.b.in; do
#             testName=$(basename $test)
#             testOutName=$(echo $testName | sed 's/s[0-9][0-9]*.//g')
#             outName=$E1ResultsDir/$solution\$${testOutName::-5}\$$(date '+%Y-%m-%d-%H-%M-%S' -d @$(stat -c %Y $solution.e)).out

#             echo "Running $(basename $outName)"

#             touch $outName
#             timeout $singleRunTimeout ./$solution.e $test /dev/null $defaultBatchSize 2>>$outName
#             echo "" >>$outName
#         done
#         echo
#     done
# fi

# if $runE2; then
#     echo "Running Experiment 2"

#     if $E2PurgeExistingTests; then
#         echo "Purging Existing Tests"
#         rm $E2TestsDir -rf
#     fi

#     echo "Generating Tests"
#     mkdir -p $E2TestsDir
#     mkdir -p $E2ResultsDir

#     for size in ${E2TestSizes[@]}; do
#         for graspSize in ${E2GraspSizes[@]}; do
#             genTest $E2TestsDir b $size $size $graspSize $E2DifferentSeeds
#         done
#     done

#     for solution in ${E2SolutionsToTest[@]}; do
#         make $solution.e
#         echo "Testing $solution"
#         for test in $E2TestsDir/*.b.in; do
#             for batch in ${E2BatchSizes[@]}; do
#                 testName=$(basename $test)
#                 testOutName=$(echo $testName | sed 's/s[0-9][0-9]*.//g')
#                 outName=$E2ResultsDir/$solution\$batch:$batch#${testOutName::-5}\$$(date '+%Y-%m-%d-%H-%M-%S' -d @$(stat -c %Y $solution.e)).out

#                 echo "Running $(basename $outName)"

#                 touch $outName
#                 timeout $singleRunTimeout ./$solution.e $test /dev/null $batch 2>>$outName
#                 echo "" >>$outName
#             done
#         done
#         echo
#     done
# fi

# if $runE3; then
#     echo "Running Experiment 3"

#     if $E3PurgeExistingTests; then
#         echo "Purging Existing Tests"
#         rm $E3TestsDir -rf
#     fi

#     echo "Generating Tests"
#     mkdir -p $E3TestsDir
#     mkdir -p $E3ResultsDir

#     for size in ${E3TestSizes[@]}; do
#         for graspSize in ${E3GraspSizes[@]}; do
#             genTest $E3TestsDir b $size $size $graspSize $E3DifferentSeeds
#         done
#     done

#     mkdir -p $E3ResultsDir

#     for solution in ${E3SolutionsToTest[@]}; do
#         make $solution.e
#         echo "Testing $solution"
#         for test in $E3TestsDir/*.b.in; do
#             testName=$(basename $test)
#             testOutName=$(echo $testName | sed 's/s[0-9][0-9]*.//g')
#             outName=$E3ResultsDir/$solution\$${testOutName::-5}\$$(date '+%Y-%m-%d-%H-%M-%S' -d @$(stat -c %Y $solution.e)).out

#             echo "Running $(basename $outName)"

#             touch $outName
#             timeout $singleRunTimeout ./$solution.e $test /dev/null $defaultBatchSize 2>>$outName
#             echo "" >>$outName
#         done
#         echo
#     done
# fi

# if $runE4; then
#     echo "Running Experiment 4"

#     if $E4PurgeExistingTests; then
#         echo "Purging Existing Tests"
#         rm $E4TestsDir -rf
#     fi

#     echo "Generating Tests"
#     mkdir -p $E4TestsDir
#     mkdir -p $E4ResultsDir

#     for size in ${E4TestSizes[@]}; do
#             genTest $E4TestsDir b $size $size -1 $E4DifferentSeeds
#     done

#     for solution in ${E4SolutionsToTest[@]}; do
#         make $solution.e
#         echo "Testing $solution"
#         for test in $E4TestsDir/*.b.in; do
#             for s in ${E4S[@]}; do
#                 testName=$(basename $test)
#                 testOutName=$(echo $testName | sed 's/s[0-9][0-9]*.//g')
#                 outName=$E4ResultsDir/$solution\$S:$s#${testOutName::-5}\$$(date '+%Y-%m-%d-%H-%M-%S' -d @$(stat -c %Y $solution.e)).out

#                 echo "Running $(basename $outName)"

#                 touch $outName
#                 timeout $singleRunTimeout ./$solution.e $test /dev/null $defaultBatchSize $s 2>>$outName
#                 echo "" >>$outName
#             done
#         done
#         echo
#     done
# fi