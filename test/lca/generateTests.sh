make generateSimple.e

seeds=("15404" "28067" "539" "15404" "166" "11160" "11400" "8758" "6462" "24751")

function genTest {
    testDir=$1
    testType=$2
    V=$3
    Q=$4
    graspSize=$5
    repeat=$6

    templName=$testDir/V:$V#Q:$Q#grasp:$graspSize

    for i in $(seq 1 $repeat); do
        name=$templName#s$i.$testType.in
        if [ ! -f $name ]; then
            echo "generating $name"

            if [ "$testType" == "b" ]; then
                ./generateSimple.e $V $Q $graspSize ${seeds[$i]} $name 
            else
                ./generateSimple.e $V $Q $graspSize ${seeds[$i]} >$name
            fi
        fi
        echo "$name generated."
    done
}

# function genText {
#     name=$testsDir/$1$(echo $2 | numfmt --to=si).$(echo $3 | numfmt --to=si).t.in
#     if [ ! -f $name ]; then
#         echo "generating $name" 
#         ./generate$1.e $2 $3 >$name
#     fi
#     echo "$name generated" 
#     echo
# }

# function genBin {
#     name=$testsDir/$1$(echo $2 | numfmt --to=si).$(echo $3 | numfmt --to=si).b.in
#     if [ ! -f $name ]; then
#         echo "generating $name" 
#         ./generate$1.e $2 $3 $name
#     fi
#     echo "$name generated" 
#     echo
# }


# for i in $(seq 1 100);
# do
#   genText Simple 100 $i
#   genBin Simple 100 $i
# done

# genText Simple 10 10
# genText Simple 9 9
# genText Simple 8 8
# genText Simple 7 7
# genText Simple 10000 10000
# genText Simple 100000 100000

# genBin Simple 10 10
# genBin Simple 1000 1000
# genBin Simple 2000 2000
# genBin Simple 3000 3000
# genBin Simple 4000 4000
# genBin Simple 5000 5000
# genBin Simple 6000 6000
# genBin Simple 7000 7000
# genBin Simple 8000 8000
# genBin Simple 9000 9000
# genBin Simple 1000000 100000
# genBin Simple 5000000 500000
# genBin Simple 10000000 1000000
# genBin Simple 15000000 1500000
# genBin Simple 20000000 2000000
# genBin Simple 25000000 2500000
# genBin Simple 30000000 3000000
# genBin Simple 35000000 3500000
# genBin Simple 40000000 4000000
# genBin Simple 45000000 4000000
# genBin Simple 50000000 5000000

# genBin LongSimple 50000000 300000
# genBin LongSimple 45000000 300000
# genBin LongSimple 40000000 300000
# genBin LongSimple 35000000 300000
# genBin LongSimple 30000000 3000
# genBin LongSimple 25000000 3000
# genBin LongSimple 20000000 3000
# genBin LongSimple 15000000 3000
# genBin LongSimple 10000000 3000
# genBin LongSimple 5000000 3000
# genBin LongSimple 1000000 3000

# echo "OK"
