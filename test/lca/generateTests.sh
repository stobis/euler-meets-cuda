make generateSimple.e
make generateScaleFree.e

seeds=("15404" "28067" "539" "15404" "166" "11160" "11400" "8758" "6462" "24751")

function genTest {
    testDir=$1
    generator=$2
    V=$3
    Q=$4
    graspSize=$5
    repeat=$6

    templName=$testDir/V_$V#Q_$Q#grasp_$graspSize

    for i in $(seq 1 $repeat); do
        name=$templName#s$i.in
        if [ ! -f $name ]; then
            echo "generating $name with $generator"

            if [ "$generator" = "simple" ]; then
                ./generateSimple.e $V $Q $graspSize ${seeds[$i]} $name 
            else
                ./generateScaleFree.e $V $Q ${seeds[$i]} $name 
            fi
        fi
        echo "$name generated."
    done
}