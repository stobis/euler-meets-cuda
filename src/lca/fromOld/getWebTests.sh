destinationDir=~/storage/webTests
mkdir $destinationDir

testsDir=~/storage/tests
mkdir $testsDir

##### Tests from http://www.cc.gatech.edu/dimacs10/archive/data/kronecker

# kronTests=(
#     "http://www.cc.gatech.edu/dimacs10/archive/data/kronecker/kron_g500-logn16.graph.bz2"
#     "http://www.cc.gatech.edu/dimacs10/archive/data/kronecker/kron_g500-logn17.graph.bz2"
#     "http://www.cc.gatech.edu/dimacs10/archive/data/kronecker/kron_g500-logn18.graph.bz2"
#     "http://www.cc.gatech.edu/dimacs10/archive/data/kronecker/kron_g500-logn19.graph.bz2"
#     "http://www.cc.gatech.edu/dimacs10/archive/data/kronecker/kron_g500-logn20.graph.bz2"
#     "http://www.cc.gatech.edu/dimacs10/archive/data/kronecker/kron_g500-logn21.graph.bz2"
# )

# kronDir=kron
# mkdir $destinationDir/$kronDir

# for test in ${kronTests[@]}; do
#     wget -nc -P $destinationDir/$kronDir $test
# done


# echo "All downloads done"

# for test in $(ls $destinationDir/$kronDir | grep .*bz2$); do
#     name=${test::-4}
#     if [ ! -f $destinationDir/$kronDir/$name ]; then
#         echo Unpacking $name
#         bzip2 -k -d $destinationDir/$kronDir/$name.bz2
#     fi
# done

# make parseDimacs.e

# for test in $(ls $destinationDir/$kronDir | grep .*graph$); do
#     if [ ! -f $testsDir/$test.b.in ]; then
#         echo "Parsing $test"
#         ./parseDimacs.e $destinationDir/$kronDir/$test $testsDir/$test.b.in 2>/dev/null
#     fi
# done


##### Tests from http://networkrepository.com

roadTests=(
    "http://nrvis.com/download/data/road/road-asia-osm.zip"
    "http://nrvis.com/download/data/road/road-belgium-osm.zip"
    "http://nrvis.com/download/data/road/road-germany-osm.zip"
    "http://nrvis.com/download/data/road/road-road-usa.zip"
    "http://nrvis.com/download/data/road/road-great-britain-osm.zip"
)

roadDir=road
mkdir $destinationDir/$roadDir

for test in ${roadTests[@]}; do
    wget -nc -P $destinationDir/$roadDir $test
done

echo "All downloads done"

for test in $(ls $destinationDir/$roadDir | grep .*zip$ ); do
    name=${test::-4}
    if [ ! -f $destinationDir/$roadDir/$name.mtx ]; then
        echo Unpacking $name
        unzip -o $destinationDir/$roadDir/$name.zip -d $destinationDir/$roadDir
    fi
done

make parseRoad.e

for test in $(ls $destinationDir/$roadDir | grep .*mtx$); do
    if [ ! -f $testsDir/$test.b.in ]; then
        echo Parsing $test
        cat $destinationDir/$roadDir/$test | grep -v ^% >$destinationDir/$roadDir/tmp
        ./parseRoad.e <$destinationDir/$roadDir/tmp $testsDir/$test.b.in 2>/dev/null
        rm $destinationDir/$roadDir/tmp
    fi
done