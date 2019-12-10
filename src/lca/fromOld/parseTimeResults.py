#!/usr/bin/python3
import os
import sys

resultsDirectory = ""
if len(sys.argv) <= 1:
    resultsDirectory = "./resultTimes/E1/"
else:
    resultsDirectory = sys.argv[1]

if not resultsDirectory.endswith("/"):
    resultsDirectory+="/"


class result:

    def __init__(self, solutionName, testName, timestamp, combined, singleResults, parameters):
        # self.solutionName = solutionName
        self.solutionName = solutionName + " (" + timestamp + ")"
        self.testName = testName
        self.timestamp = timestamp
        self.combined = combined
        self.singleResults = singleResults
        self.parameters = parameters

        numOfQueries = 0
        combinedQueryTimes = 0.0
        for s in singleResults:
            numOfQueries += s.numOfQueries
            combinedQueryTimes += s.combinedQueryTimes

        if (numOfQueries > 0):
            self.oneQueryTime = combinedQueryTimes / numOfQueries
            self.combined.append(("AvgQueryTime(ns)", self.oneQueryTime * 10**9))
        else:
            self.oneQueryTime = 0

    # def __repr__(self):
    #     return "Solution: "+self.solutionName+", on: "+self.testName+". Timestamp: "+self.timestamp+"\n"+str(self.combined)


class singleResult:  # whole

    def __init__(self, rawSingleResult):
        self.sectionNames = []
        self.sectionTimes = []

        self.numOfQueries = 0
        self.combinedQueryTimes = 0.0

        res = rawSingleResult.split("\n")
        for line in res:
            if "Whole" in line:
                sectionName, time, tmp = line.split(",")
                self.sectionNames.append(sectionName)
                self.sectionTimes.append(float(time))
                if "Queries" in line:
                    self.combinedQueryTimes += float(time)
            if "NumOfQueries" in line:
                sectionName, time, q = line.split(",")
                self.numOfQueries += int(q.split(":")[1])


def getResultFromFilename(filename):
    print(filename)
    solutionName, testName, timestamp = filename[:-4].split("$")
    testProperties = testName.split("#")
    testProperties[:] = [prop.split(":") for prop in testProperties]

    testProperties = {prop[0]: prop[1] for prop in testProperties}
    print(testProperties)

    content = []
    with open(resultsDirectory + filename) as rawResult:
        content.extend(rawResult.read().split("\n\n"))

    singleResults = []

    for res in content:
        if len(res) > 1:
            singleResults.append(singleResult(res))

    results = []

    names = singleResults[0].sectionNames
    times = []

    for res in singleResults:
        for i in range(len(res.sectionNames)):
            if len(times) <= i:
                times.append([])
            times[i].append(res.sectionTimes[i])

    for i in range(len(names)):
        time = 0.0
        for t in times[i]:
            time += t

        time /= len(times[i])
        results.append((names[i], time))

    return result(solutionName, testName, timestamp, results, singleResults, testProperties)


resultsRaw = []
for (dirpath, dirnames, filenames) in os.walk(resultsDirectory):
    resultsRaw.extend(filenames)
    break

results = []
for filename in resultsRaw:
    if filename[0] != '.':
        results.append(getResultFromFilename(filename))

testNames = []
for res in results:
    if not res.testName in testNames:
        testNames.append(res.testName)

solutions = []
for res in results:
    if not res.solutionName in solutions:
        solutions.append(res.solutionName)

for testName in testNames:
    print("Results on " + testName)

    for solution in solutions:
        print(" " + solution + ":")

        for res in results:
            if res.solutionName == solution and res.testName == testName:
                print("   (timestamp: " + res.timestamp)
                for c in res.combined:
                    print('   {:<18}{:>18}ms'.format(c[0], str(round(c[1], 4))))

                print('   {:<18}{:>18}ns'.format("AverageQueryTime", round(res.oneQueryTime * 10**9, 4)))
                print()
    print()


experimentName=os.path.basename(os.path.dirname(resultsDirectory))

if experimentName=="E1":
    testsParameters = ["V", "V"]
    testsConstraints = [{"grasp": -1}, {"grasp":1000}]
    # testsConstraints = [{"grasp": 10}]
    testsParametersValues = []
elif experimentName=="E2":
    testsParameters = ["batch"]
    testsConstraints = [{}]
    testsParametersValues = []
elif experimentName=="E3":
    testsParameters = ["grasp"]
    testsConstraints = [{}]
    testsParametersValues = []
elif experimentName=="E4":
    # testsParameters = ["V", "V", "V", "V", "V", "V"]
    # testsConstraints = [{"S":800},{"S":1600},{"S":3200},{"S":6400},{"S":12800},{"S":25600},]
    testsParameters = ["S", "S", "S", "S", "S", "S", "S"]
    testsConstraints = [{"V":1000000},{"V":2000000},{"V":4000000},{"V":8000000},{"V":16000000},{"V":32000000},{"V":64000000},]
    testsParametersValues = []
else:
    print("Only E1, E2, E3 supported for now")
    exit(1)
    # testsParameters = ["grasp"]
    # testsConstraints = [{"V": 1000000}]
    # testsParametersValues = []


for i, name in enumerate(testsParameters):
    while len(testsParametersValues) <= i:
        testsParametersValues.append([])
    for res in results:
        if name in res.parameters:
            if res.parameters[name] not in testsParametersValues[i]:
                testsParametersValues[i].append(res.parameters[name])

print("Results of experiment: " + experimentName)

sectionNames = ["Preprocessing", "AvgQueryTime(ns)", "Queries"]
# sectionNames=["LR1", "LR2"]
# sectionNames = ["Preprocessing", "AvgQueryTime(ns)", "List Rank"]

for i, param in enumerate(testsParameters):
    print("Results by " + param+",")
    if len(testsConstraints[i]) > 0:
        print("Test Constraints: "+str(testsConstraints[i]))
    for sectionName in sectionNames:
        print(sectionName +",", end="")
        for paramValue in testsParametersValues[i]:
            print(paramValue+",", end="")
        print()
        for solution in solutions:
            print(solution + ",", end="")
            for paramValue in testsParametersValues[i]:
                for testName in testNames:
                    for res in results:
                        if (res.testName == testName and res.parameters[param] == paramValue and
                                res.solutionName == solution):
                            isOk=True
                            for c in testsConstraints[i]:
                                if res.parameters[c] != str(testsConstraints[i][c]):
                                    isOk=False
                            if isOk==True:
                                for sectionRes in res.combined:
                                    if (sectionRes[0] == sectionName):
                                        print(str(round(sectionRes[1], 4)) + ",", end="")

            print()
        print()
            
            # print(paramValue + ":")
            #     for testName in testNames:
            #         for res in results:
            #             if (res.testName == testName and res.parameters[param] == paramValue):
            #                 print(res.testName + ",", end="")
            #                 break

            #     print()

            #     for solution in solutions:
            #         print(solution + ",", end="")
            #         for testName in testNames:
            #             for res in results:
            #                 if (res.testName == testName and res.parameters[param] == paramValue and
            #                         res.solutionName == solution):
            #                     for sectionRes in res.combined:
            #                         if (sectionRes[0] == sectionName):
            #                             print(str(round(sectionRes[1], 4)) + ",", end="")
            #         print()
            #     print()

# for testPrefix in testPrefixes:
#     print(testPrefix)
#     for sectionName in sectionNames:
#         print(sectionName + ",", end="")
#         for testName in testNames:
#             if testName.startswith(testPrefix):
#                 print(testName + ",", end="")

#         print()

#         for solution in solutions:
#             print(solution + ",", end="")
#             for res in results:
#                 for testName in testNames:
#                     if testName.startswith(
#                             testPrefix
#                     ) and res.solutionName == solution and res.testName == testName:
#                         for sectionRes in res.combined:
#                             if (sectionRes[0] == sectionName):
#                                 print(
#                                     str(round(sectionRes[1], 4)) + ",", end="")
#             print()

#         print()

#     print()

# print(results)
