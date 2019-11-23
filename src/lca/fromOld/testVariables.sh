testsDir=$(realpath ~/storage/tests)
resultTimesDir=resultTimes

defaultBatchSize=-1

singleRunTimeout=300

validityTestsDir=$testsDir/validity
validityAnswersDir=$testsDir/validityOut
validityTestsSizes=(
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
    1000000
    2000000
)
validityGraspSize=-1
validityOutGenerator="cpuRmqLCA"
# validitySolutionToTest="cudaInlabelLCA"
validitySolutionToTest="cpuInlabelLCA"
# validitySolutionToTest="cudaSimpleLCA"

runE1=true
# runE1=false
# E1PurgeExistingTests=false
E1PurgeExistingTests=true
E1SolutionsToTest=(
    "cudaInlabelLCA"
    "cudaSimpleLCA"
    # "cpuRmqLCA"
    "cpuInlabelLCA"
)
E1TestsDir=$testsDir/E1
E1ResultsDir=$resultTimesDir/E1
E1TestSizes=(
    1000000
    2000000
    # 4000000
    # 8000000
    # 16000000
    # 32000000
    # 64000000
)
E1GraspSizes=( #how far up a father can be
    -1
    1000
)
E1DifferentSeeds=2

runE2=true
# runE2=false
# E2PurgeExistingTests=true
E2PurgeExistingTests=false
E2SolutionsToTest=(
    "cudaInlabelLCA"
    "cpuRmqLCA"
    "cpuInlabelLCA"
)
E2TestsDir=$testsDir/E2
E2ResultsDir=$resultTimesDir/E2
E2TestSizes=(
    8000000
)
E2GraspSizes=(
    -1
)
E2DifferentSeeds=2
E2BatchSizes=(
    # 1
    # 10
    # 100
    1000
    10000
    100000
    1000000
    10000000
    100000000
)

# runE3=false
runE3=true
E3PurgeExistingTests=false
E3SolutionsToTest=(
    "cudaInlabelLCA"
    # "cudaSimpleLCA"
    "cpuInlabelLCA"
)
E3TestsDir=$testsDir/E3
E3ResultsDir=$resultTimesDir/E3
E3TestSizes=(
    8000000
)
E3GraspSizes=(
    # 1
    # 10
    # 100
    # 1000
    # 10000
    100000
    1000000
    10000000
)
E3DifferentSeeds=2


# # runE4=true
runE4=false
# # E2PurgeExistingTests=true
# E4PurgeExistingTests=false
# E4SolutionsToTest=(
#     "cudaInlabelLCA"
# )
# E4TestsDir=$testsDir/E4
# E4ResultsDir=$resultTimesDir/E4
# E4TestSizes=(
#     1000000
#     2000000
#     4000000
#     8000000
#     16000000
#     32000000
#     64000000
# )
# E4S=(
#     800
#     1600
#     3200
#     6400
#     12800
#     25600
# )
# E4DifferentSeeds=5




generateAnswers=true


repeatSingleTest=10

progressBarWidth=50
