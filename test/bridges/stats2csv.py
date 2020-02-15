#!/usr/bin/python3

import csv
import sys

data_dict = {}

def parse_one(first, input):
    first = first[4:]
    first = first.split(':')
    test_name = first[1][1:]
    data_dict[test_name] = {}
    # print(data_dict)
    while True:
        line = input.readline()
        if not line:
            print('WARN: Last dataset is incomplete')
            del data_dict[test_name]
            return
        elif line.startswith('%%% #'):
            line = line.split(':')
            data_dict[test_name][line[0][4:]] = int(line[1])
            return
        elif line.startswith('%%% N') or line.startswith('%%% M'):
            line = line.split(':')
            var = line[0][-1:]
            data_dict[test_name][var] = int(line[1])
            continue
        elif line.startswith('%%%'):
            print('WARN: Invalid dataset ' + test_name + ' (num of bridges is unknown)')
            del data_dict[test_name]
            line  = line[:-1]
            # print(line)
            parse_one(line, input)
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
        data_dict[test_name][algo].append((param,time))

with open(sys.argv[1], 'r', newline='') as datafile:
    while True:
        line = datafile.readline() 
        if not line:
            break
        line = line[:-1]
        # print(line)

        if line.startswith('%%% File'):
            parse_one(line, datafile)
        
# print(data_dict)

fieldnames_dict = {}

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
        fieldnames = ['file', 'N', 'M', '# bridges', 'algo'] + fieldnames
        # print(fieldnames)

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for (filename, fileinfo) in sorted(data_dict.items()):
            row = {}
            row['file'] = filename
            row['N'] = fileinfo['N']
            row['M'] = fileinfo['M']
            row['# bridges'] = fileinfo['# Bridges']
            row['algo'] = algo
            # print(row)
            # print(fileinfo[algo])
            for (name, val) in fileinfo[algo]:
                row[name] = val
            # print(row)
            writer.writerow(row)
