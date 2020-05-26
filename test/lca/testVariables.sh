testsDir=tests
resultTimesDir=resultTimesTemporary
defaultBatchSize=-1
runnerPath="../../lca_runner.e"
singleRunTimeout=60


### Validity tests

validityTestsDir=$testsDir/validity
validityAnswersDir=$testsDir/validityOut

validityTestsSizes=(
    5
    10
    1000
    2000
    3000
    4000
    5000
    6000
    7000
    8000
    9000
    100000
    500000
    1000000
    2000000
    # 3000000
    # 4000000
    # 10000000
)
validityGraspSize=-1
validityOutGeneratorAlgorithm="cpu-rmq"
validitySolutionsToTest=(
    cuda-inlabel
    cuda-naive
    cpu-inlabel
    cpu-simple
)

### Time tests

generateAnswers=true

testsToRun=(
    1
    2
    3
)

## Experiment 1 - all algos, different graph sizes, shallow (grasp -1) and mid-deep (grasp 1000)
E1SolutionsToTest=(
    "cuda-inlabel"
    "cuda-naive"
    # "cpu-rmq"
    "cpu-inlabel"
)
E1TestsDir=$testsDir/E1
E1ResultsDir=$resultTimesDir/E1
E1TestSizes=(
    1000000
    2000000
    4000000
    8000000
    # 16000000
    # 32000000
    # 64000000
)
E1GraspSizes=( #how far up a father can be
    -1
    1000
)
E1BatchSizes=(
    -1
)
E1DifferentSeeds=5

## Experiment 2 - CUDA inlabel by batch size
E2SolutionsToTest=(
    "cuda-inlabel"
    "cpu-inlabel"
)
E2TestsDir=$testsDir/E2
E2ResultsDir=$resultTimesDir/E2
E2TestSizes=(
    1000000
)
E2GraspSizes=(
    -1
)
E2BatchSizes=(
    1
    10
    100
    1000
    10000
    100000
    1000000
    # 10000000
    # 100000000
)
E2DifferentSeeds=5

E3SolutionsToTest=(
    "cuda-inlabel"
    "cuda-naive"
    # "cpuInlabelLCA"
)

## Experiment 3 - CUDA inlabel vs naive by grasp size
E3TestsDir=$testsDir/E3
E3ResultsDir=$resultTimesDir/E3
E3TestSizes=(
    8000000
)
E3GraspSizes=(
    # 1
    # 10
    # 100
    1000
    10000
    100000
    1000000
    10000000
)
E3BatchSizes=(
    -1
)
E3DifferentSeeds=5