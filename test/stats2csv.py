#!/usr/bin/python3

import csv
import sys

data_dict = {}
problem = "Unk"


def parse_one(first, input):
    first = first[4:]
    first = first.split(':')
    global problem
    problem = first[0]
    test_name = first[2][1:]
    data_dict[test_name] = {}

    # Lca encodes test properties in test filename.
    test_metadata = test_name.split('$')
    if len(test_metadata) > 1:
        test_vars = test_metadata[1].split('#')
        for var in test_vars:
            var = var.split('_')
            if len(var) > 1:
                data_dict[test_name][var[0]] = var[1]

    # print(data_dict)
    while True:
        line = input.readline()
        if not line:
            # print('WARN: Last dataset is incomplete')
            return
        elif line.startswith('%%% #') or line.startswith('%%% MaxHeight') or line.startswith("%%% numQ"):
            line = line.split(':')
            data_dict[test_name][line[0][4:]] = int(line[1])
            continue
        elif line.startswith('%%% AvgHeight'):
            line = line.split(':')
            data_dict[test_name][line[0][4:]] = float(line[1])
            continue
        elif line.startswith('%%% N') or line.startswith('%%% M'):
            line = line.split(':')
            var = line[0][-1:]
            data_dict[test_name][var] = int(line[1])
            continue
        elif line.startswith('%%%'):
            continue
            # print('WARN: Invalid dataset ' + test_name +
            #       ' (num of bridges is unknown)')
            # del data_dict[test_name]
            # line = line[:-1]
            # print(line)
            # parse_one(line, input)
            # return
        if line.isspace():
            return
        line = line.split(':')
        algo = line[0].strip()
        param = line[1].strip()
        time = line[2].strip()
        if time.endswith(' ms.'):
            time = time[:-4]
        # print(algo, param, time)
        if algo not in data_dict[test_name]:
            data_dict[test_name][algo] = []
        data_dict[test_name][algo].append((param, time))


with open(sys.argv[1], 'r', newline='') as datafile:
    while True:
        line = datafile.readline()
        if not line:
            break
        line = line[:-1]
        # print(line)

        if line.startswith('%%% Bridges: File') or line.startswith('%%% Lca: File'):
            parse_one(line, datafile)

# print(data_dict)

fieldnames_dict = {}

if problem != "Bridges" and problem != "Lca":
    print("Unknown Problem. Problem on first line: " + problem)

for (test, test_info) in data_dict.items():
    # print('test: ' + test)
    for key in test_info.keys():
        if type(test_info[key]) is not list:
            continue
        # algo = key
        timestamps = test_info[key]
        # print(timestamps)
        fieldnames_dict[key] = [k for (k, v) in timestamps]

# print(fieldnames_dict)

with open(sys.argv[2], 'w', newline='') as csvfile:
    for (algo, fieldnames) in fieldnames_dict.items():
        # print('algo: ' + algo)
        if problem == "Bridges":
            fieldnames = ['file', 'N', 'M', '# bridges', 'algo'] + fieldnames
        elif problem == "Lca":
            fieldnames = ['file', 'N', 'numQ', 'algo',
                          'grasp', 'batch', 'AvgHeight', 'MaxHeight'] + fieldnames
        # print(fieldnames)

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for (filename, fileinfo) in sorted(data_dict.items()):
            row = {}
            if problem == "Bridges":
                row['file'] = filename
                row['N'] = fileinfo['N']
                row['M'] = fileinfo['M']
                row['# bridges'] = fileinfo['# Bridges']
                row['algo'] = algo
            elif problem == "Lca":
                row['file'] = filename
                row['N'] = fileinfo['N']
                row['numQ'] = fileinfo['numQ']
                row['algo'] = algo
                row['grasp'] = fileinfo['grasp']
                row['batch'] = fileinfo['batch']
                row['AvgHeight'] = fileinfo['AvgHeight']
                row['MaxHeight'] = fileinfo['MaxHeight']
            # print(row)
            # print(fileinfo[algo])
            for (name, val) in fileinfo[algo]:
                row[name] = val
            # print(row)
            writer.writerow(row)
