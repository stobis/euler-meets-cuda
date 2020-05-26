OPTIND=1  # Reset in case getopts has been used previously in the shell.

testDirectory="test/lca"
tmpCsvFolder=tmpcsv

__help="
Usage: lca_test.sh [-h | -c | -t | -r]

Options:
  -h        Show this help and exit
  -c        Check all algorithms. Runs all algorithms on a set of tests and checks if all outputs are same
  -t \$out   Run timed tests, save output to \$out.
  -r \$times Repeat tests \$times times and get average results. Default value is 1.
"

run_check=0
run_time=0
repeats=1

while getopts "ht:cr:" opt; do
    case "$opt" in
    h)
        echo "$__help"
        exit 0
        ;;
    c)  run_check=1
        ;;
    t)  run_time=1
        resultsFile=$(realpath $OPTARG)
        ;;
    r)  repeats=$OPTARG
        ;;
    esac
done

if [ "$run_check" = 0 ] && [ "$run_time" = 0 ]; then
  echo "$__help"
  exit
fi

if [ "$run_check" = 1 ]; then
  make lca_runner
  if [ $? -ne 0 ]; then
    echo "lca_runner build FAILED!"
    exit 1
  fi
  echo "Running valitity checks."
  cwd=$(pwd)
  cd $testDirectory
  ./testSolution.sh
  cd $cwd
fi

if [ "$run_time" = 1 ]; then
  if [ -z "$resultsFile" ]; then
    echo "$__help"
    exit
  fi

  make lca_runner
  if [ $? -ne 0 ]; then
    echo "lca_runner build FAILED!"
    exit 1
  fi
  echo "Running timed tests."
  cwd=$(pwd)
  cd $testDirectory
  
  mkdir $tmpCsvFolder
  for ((i=0;i<$repeats;i++))
  do
      ./testTime.sh $tmpCsvFolder/$i
  done
  ../combine_csv.py $(pwd)/$tmpCsvFolder $resultsFile
  rm -rf $tmpCsvFolder
  
  cd $cwd
fi
