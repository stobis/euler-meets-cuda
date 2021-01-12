OPTIND=1  # Reset in case getopts has been used previously in the shell.

testDirectory="test/lca"
tmpCsvFolder=tmpcsv

__help="
Usage: lca_test.sh [-h | -c | -t | -r]

Options:
  -h        Show this help and exit
  -c [algorithm]       Check algorithms. The algorithm parameter is optional, if not provided runs all algorithms. Tests the algorithm(s) and compares output to an algoririthm known to be correct. Run ./lca_runner.e -h to get list of available algorithms.
  -t \$out   Run timed tests, save output to \$out.
  -r \$times Repeat tests \$times times and get average results. Default value is 1.
  -p        Generate plots in addition to .csv file.
"

run_check=0
run_time=0
repeats=1
gen_plot=0

while getopts "ht:cr:p" opt; do
    case "$opt" in
    h)
        echo "$__help"
        exit 0
        ;;
    c)  run_check=1
        ;;
    t)  run_time=1
        resultsFileFinal=$(realpath $OPTARG)
        ;;
    r)  repeats=$OPTARG
        ;;
    p)  gen_plot=1
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
  if [ -z "$resultsFileFinal" ]; then
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
      . testTime.sh $tmpCsvFolder/$i
  done
  if ../combine_csv.py $(pwd)/$tmpCsvFolder $resultsFileFinal; then
    rm -rf $tmpCsvFolder
  fi
  
  cd $cwd

  if [ "$gen_plot" = 1 ]; then
    python ./test/plot.py $resultsFileFinal
  fi
fi
