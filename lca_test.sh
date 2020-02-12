OPTIND=1  # Reset in case getopts has been used previously in the shell.

testDirectory="test/lca"

__help="
Usage: lca_test.sh [-h | -c | -t | -v]

Options:
  -h      Show this help and exit
  -c      Check all algorithms. Runs all algorithms on a set of tests and checks if all outputs are same
  -t      Run timed tests
  -v      Verbose
"

verbose=0
run_check=0
run_time=0

while getopts "hvtc" opt; do
    case "$opt" in
    h)
        echo "$__help"
        exit 0
        ;;
    v)  verbose=1
        ;;
    c)  run_check=1
        ;;
    t)  run_time=1
        ;;
    esac
done

if [ "$run_check" = 0 ] && [ "$run_time" = 0 ]; then
  echo "$__help"
  exit
fi

if [ "$run_check" = 1 ]; then
  make lca_runner
  echo "Running valitity checks."
  cwd=$(pwd)
  cd $testDirectory
  ./testSolution.sh
  cd $cwd
fi

if [ "$run_time" = 1 ]; then
  make lca_runner
  echo "Running timed tests."
  cwd=$(pwd)
  cd $testDirectory
  ./testTime.sh
  cd $cwd
fi