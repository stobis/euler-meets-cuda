testsDir=$1

analyzerName=testStats.e

make $analyzerName

for i in $(ls $testsDir/*.b.in); do
    i=$(basename $i)
    # echo "Generating stats for $i"
    ./$analyzerName $testsDir/$i 
    echo ""
done
