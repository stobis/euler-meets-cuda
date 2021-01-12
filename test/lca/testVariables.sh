testsDir=tests
resultTimesDir=resultTimesTemporary
defaultBatchSize=-1
runnerPath="../../lca_runner.e"
singleRunTimeout=120


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
    3000000
    4000000
    10000000
    20000000
    30000000
)
validityGraspSize=-1
validityOutGeneratorAlgorithm="cpu-rmq"
validitySolutionsToTest=(
    cuda-inlabel
    cuda-naive
    cpu-inlabel
    # cpu-simple
    multicore-cpu-inlabel
)

### Time tests

generateAnswers=true

testsToRun=(
    1
    2
    3
    4
    5
)

## Experiment 1 - all algos, different graph sizes, shallow (grasp -1) and mid-deep (grasp 1000)
E1SolutionsToTest=(
    "cuda-inlabel"
    "cuda-naive"
    # "cpu-rmq"
    "cpu-inlabel"
    "multicore-cpu-inlabel"
)
E1TestsDir=$testsDir/E1
E1ResultsDir=$resultTimesDir/E1
E1TestSizes=(
    1000000
    2000000
    4000000
    8000000
    16000000
    32000000
)
E1GraspSizes=( #how far up a father can be
    -1
    1000
)
E1BatchSizes=(
    -1
)
E1DifferentSeeds=5
E1Generator=simple
E1CheckAnswers=true

## Experiment 2 - CUDA inlabel by batch size
E2SolutionsToTest=(
    "cuda-inlabel"
    "cpu-inlabel"
    "multicore-cpu-inlabel"
)
E2TestsDir=$testsDir/E2
E2ResultsDir=$resultTimesDir/E2
E2TestSizes=(
    8000000
)
E2NumQueries=(
    10000000
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
    10000000
)
E2DifferentSeeds=5
E2Generator=simple
E2CheckAnswers=true

## Experiment 3 - CUDA inlabel vs naive by grasp size
E3SolutionsToTest=(
    "cuda-inlabel"
    "cuda-naive"
    "multicore-cpu-inlabel"
)
E3TestsDir=$testsDir/E3
E3ResultsDir=$resultTimesDir/E3
E3TestSizes=(
    8000000
)
E3GraspSizes=(
    1
    3
    10
    33
    100
    333
    1000
    3333
    10000
    33333
    100000
    333333
    1000000
    10000000
)
E3BatchSizes=( 
    -1
)
E3DifferentSeeds=5
E3Generator=simple
E3CheckAnswers=true

## Experiment 4 - CUDA inlabel vs naive by num of queries
E4SolutionsToTest=(
    "cuda-inlabel"
    "cuda-naive"
    "multicore-cpu-inlabel"
)
E4TestsDir=$testsDir/E4
E4ResultsDir=$resultTimesDir/E4
E4TestSizes=(
    8000000
)
E4GraspSizes=(
   -1
   1000
)
E4BatchSizes=(
    -1
)
E4DifferentSeeds=5
E4NumQueries=(
    1000000
    2000000
    4000000
    8000000
    16000000
    32000000
    64000000
    128000000
)
E4Generator=simple
E4CheckAnswers=true


## Experiment 5 - Scale-Free
E5SolutionsToTest=(
    "cuda-inlabel"
    "cuda-naive"
    "cpu-rmq"
    "cpu-inlabel"
    "multicore-cpu-inlabel"
)
E5TestsDir=$testsDir/E5
E5ResultsDir=$resultTimesDir/E5
E5TestSizes=(
    1000000
    2000000
    4000000
    8000000
    16000000
    32000000
)
E5GraspSizes=( #irrelevant here
    -1
)
E5BatchSizes=( #irrelevant here
    -1
)
E5DifferentSeeds=5
E5Generator=scaleFree
E5CheckAnswers=true