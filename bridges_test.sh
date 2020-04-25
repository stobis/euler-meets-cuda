OPTIND=1  # Reset in case getopts has been used previously in the shell

testDirectory="test/bridges"

__help="
Usage: bridges_test.sh [-h | -t | -r]

Options:
  -h        Show this help and exit
  -t \$out   Run timed tests, save output to \$out.
  -r \$times Repeat tests \$times times and get average results. Default value is 1.
"
# TODO do we want to have some tests to check correctness?

run_time=1
repeats=1

while getopts "ht:r:" opt; do
    case "$opt" in
    h)
        echo "$__help"
        exit 0
        ;;
    t)  outFile=$(realpath $OPTARG)
        ;;
    r)  repeats=$OPTARG
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
