. testVariables.sh 

. generateTests.sh

statsPath=../stats2csv.py
tmpCsv=tmp.csv
resultsFile=$1

for T in ${testsToRun[@]}; do
    echo "Running Experiment $T"

    echo "Generating Tests"
    _testsDir=E${T}TestsDir
    mkdir -p ${!_testsDir}

    eval _testSizes=( '"${E'${T}'TestSizes[@]}"' )
    eval _graspSizes=( '"${E'${T}'GraspSizes[@]}"' )
    _numOfSeeds=E${T}DifferentSeeds
    for size in ${_testSizes[@]}; do
        for graspSize in ${_graspSizes[@]}; do
            genTest ${!_testsDir} b $size $size $graspSize ${!_numOfSeeds}
        done
    done


    _resultsDir=E${T}ResultsDir
    mkdir -p ${!_resultsDir} 

    eval _solutionsToTest=( '"${E'${T}'SolutionsToTest[@]}"' )
    for solution in ${_solutionsToTest[@]}; do
        echo "Testing $solution"
        for test in ${!_testsDir}/*.b.in; do
            testName=$(basename $test)
            testOutName=$(echo $testName | sed 's/s[0-9][0-9]*.//g') 
            outName=${!_resultsDir}/$solution\$${testOutName::-5}\$$(date '+%Y-%m-%d-%H-%M-%S' -d @$(stat -c %Y $runnerPath)).out
            mv $test $outName.test

            echo -e "\tRunning $(basename $outName)"

            timeout $singleRunTimeout $runnerPath -i $outName.test -o /dev/null -a $solution >$outName
            echo "" >>$outName

            $statsPath $outName $tmpCsv

            if [ ! -f "$resultsFile" ]; then
                cat $tmpCsv >$resultsFile
            else
                cat $tmpCsv | tail -n +2 >>$resultsFile
            fi
            rm $tmpCsv
            rm $outName
            mv $outName.test $test
        done
    done

    mkdir -p ${!_resultsDir} 

done