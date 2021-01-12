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
    eval _batchSizes=( '"${E'${T}'BatchSizes[@]}"' )
    eval _numQueries=( '"${E'${T}'NumQueries[@]}"' )


    _numOfSeeds=E${T}DifferentSeeds
    _generator=E${T}Generator

    for size in ${_testSizes[@]}; do
        for graspSize in ${_graspSizes[@]}; do
            if [ -z "$_numQueries" ]; then
                genTest ${!_testsDir} ${!_generator} $size $size $graspSize ${!_numOfSeeds}
            else
                for queries in ${_numQueries[@]}; do
                    genTest ${!_testsDir} ${!_generator} $size $queries $graspSize ${!_numOfSeeds}
                done
            fi
        done
    done

    _checkAns=E${T}CheckAnswers
    if [ "${!_checkAns}" = true ] ; then
        _outDir=E${T}TestsDir
        outDir=${!_outDir}/outs
        mkdir -p $outDir
        echo "Generating answers for E${T}"
        for test in ${!_testsDir}/*.in; do
            testName=$(basename $test)
            outName=$outDir/$testName.out

            if [ ! -f $outName ]; then
                echo "Generating $outName answer"
                $runnerPath -i $test -o $outName -a $validityOutGeneratorAlgorithm 2>/dev/null >/dev/null

                if [ ! -s $outName ]; then
                    echo "Error generating answers! $outName is empty!"
                    exit 1
                fi
                echo "$outName generated."
            fi
        done
        echo "Done generating answers"
    fi

    _resultsDir=E${T}ResultsDir
    mkdir -p ${!_resultsDir} 

    eval _solutionsToTest=( '"${E'${T}'SolutionsToTest[@]}"' )
    for solution in ${_solutionsToTest[@]}; do
        echo "Testing $solution"
        for test in ${!_testsDir}/*.in; do
            for batchSize in ${_batchSizes[@]}; do
                testName=$(basename $test)
                testOutName=$(echo $testName | sed 's/s[0-9][0-9]*.//g') 
                outName=${!_resultsDir}/$solution\$${testOutName::-3}\#batch_$batchSize\#\$$(date '+%Y-%m-%d-%H-%M-%S' -d @$(stat -c %Y $runnerPath)).out
                mv $test $outName.test

                echo -e "\tRunning $(basename $outName)"
                timeout $singleRunTimeout $runnerPath -i $outName.test -o tmp.out -a $solution >$outName -b $batchSize
                timeout_status=$?
                echo "" >>$outName

                if [ "${!_checkAns}" = true ] ; then
                    if [[ $timeout_status -eq 124 ]]; then
                        echo -e "\t\tTimed out. Skipping ans check."
                    else
                        properAns=$outDir/$testName.out
                        if diff tmp.out $properAns >/dev/null; then
                            echo -e "\t\tAns check OK"
                        else
                            echo ""
                            echo "  Wrong answer on $testName! Aborting."
                            exit 1
                        fi
                    fi
                fi
                rm tmp.out 2>/dev/null

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
    done

    mkdir -p ${!_resultsDir} 

done