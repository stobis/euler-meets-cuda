#!/usr/bin/python3

import csv
import sys
import os
from collections import defaultdict
import pprint
import numpy as np

current_header = None
current_header_str = None
# data[header][file][algo]
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for entry in os.scandir(sys.argv[1]):
    if (entry.is_file() == False):
        continue
    # print(entry)
    # print(entry.path)

    with open(entry.path) as csvfile:
        reader = csv.reader(csvfile)

        for line in reader:
            # print(line)
            if (line[0] == 'file'):
                current_header = line
                current_header_str = ",".join(line)
                # print(current_header_str)
                continue
            data[current_header_str][line[0]][line[current_header.index('algo')]].append(line)

# pp = pprint.PrettyPrinter(indent=1)
# pp.pprint(data)

with open(sys.argv[2], 'w', newline='') as csvfile:
    fieldnames = ['first_name', 'last_name']
    writer = csv.writer(csvfile)

    for header in sorted(data.keys()):
        writer.writerow(header.split(',') + ['Standard deviation'])
        # print(header)
        for file in sorted(data[header].keys()):
            for algo in sorted(data[header][file].keys()):
                results = data[header][file][algo]
                # count avg
                row = [float(0)] * len(results[0])
                for i in range(0, len(results[0])):
                    for result in results:
                        try:
                            row[i] += float(result[i])
                        except Exception:
                            row[i] = result[i]
                for i in range(0, len(row)):
                    if type(row[i]) is float:
                        row[i] /= len(results)
                # count deviation
                dev = np.std(np.array([float(x[-1]) for x in results]))
                writer.writerow(row + [dev])
