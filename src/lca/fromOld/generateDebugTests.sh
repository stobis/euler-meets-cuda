. generateTests.sh

mkdir tests

genTest tests t 10 10 -1 1
genTest tests t 10000 10000 -1 1

make cpuInlabelLCA.e
for test in tests/*.t.in; do
  ./cpuInlabelLCA.e <$test >$test.out
done