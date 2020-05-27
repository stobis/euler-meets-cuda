OPTIND=1  # Reset in case getopts has been used previously in the shell

testDirectory="test/bridges"

__help="
Usage: bridges_test.sh [-h | -t | -r | -p]

Options:
  -h        Show this help and exit
  -t \$out   Run timed tests, save output to \$out.
  -r \$times Repeat tests \$times times and get average results. Default value is 1.
  -p        Generate plots in addition to .csv file.
"
# TODO do we want to have some tests to check correctness?

run_time=1
repeats=1
gen_plot=0

while getopts "ht:r:p" opt; do
    case "$opt" in
    h)
        echo "$__help"
        exit 0
        ;;
    t)  outFile=$(realpath $OPTARG)
        ;;
    r)  repeats=$OPTARG
        ;;
    p)  gen_plot=1
        ;;
    esac
done

if [ -z "$outFile" ]; then
  echo "$__help"
  exit
fi

make bridges_runner
if [ $? -ne 0 ]; then
  echo "bridges_runner build FAILED!"
  exit 1
fi

cwd=$(pwd)

echo "Preparing timed tests."
cd $testDirectory
make all
if [ $? -ne 0 ]; then
  echo "Test parses build FAILED!"
  cd $cwd
  exit 1
fi
./prepare.py

echo "Running timed tests."
./run_experiments.sh all $repeats $outFile

cd $cwd

if [ "$gen_plot" = 1 ]; then
  python ./test/plot.py $outFile
fi