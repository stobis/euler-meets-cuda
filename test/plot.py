import matplotlib as mpl
mpl.use('Agg')

import re
import os
import sys
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

csv_filename = "file"
csv_N = 'N'
csv_algo = 'algo'
csv_overall = 'Overall'
csv_grasp = 'grasp'
csv_batch = 'batch'

csv_parameters = [csv_N, csv_grasp, csv_batch]
csv_ints = []
csv_ints.extend(csv_parameters)
csv_floats = []
csv_floats.append(csv_overall)


# Here we declare what tests consist an experiment.
# An experiment is described by a regex that should be contained in input's filename
experiments2 = [
    {"restrictions": [(csv_filename, "E1"), (csv_grasp, -1)],
     "x_param": csv_N, "x_show": csv_N, "type": "lines"},
    {"restrictions": [(csv_filename, "E1"), (csv_grasp, 1000)],
     "x_param": csv_N, "x_show": csv_N, "type": "lines"},
    {"restrictions": [(csv_filename, "E2")],
     "x_param": csv_batch, "x_show": csv_batch, "type": "lines"},
    {"restrictions": [(csv_filename, "E3")],
     "x_param": csv_grasp, "x_show": csv_grasp, "type": "lines"},
    {"restrictions": [(csv_filename, "kron")],
     "x_param": csv_N, "x_show": csv_N, "type": "lines"},
    {"restrictions": [(csv_filename, "road-g|road-a|road-r|road-b|roadNet")],
     "x_param": csv_N, "x_show": csv_filename, "type": "bars"},
]

experiments = ['E1', 'E2', 'E3',
               'USA-road', 'citationCiteseer|Patents|coPapersDBLP|delaunay_n24|in-2004|internet',
               'kron', 'roadNet|osm', 'com-LiveJournal|hollywood|Stanford',
               'road-g|road-a|road-r|road-b|roadNet', 'soc|wikipedia2009|ca-hollywood-2009',
               'amazon0601|as-skitter|soc-LiveJournal1'
               ]
experiments_use_filename = [False, False, False,
                            True, True,
                            True, True, True,
                            True, True,
                            True
                            ]


with open(sys.argv[1], 'r') as file:
    csv_reader = csv.DictReader(file)
    csv_file = []
    for line in csv_reader:
        if line[csv_filename] == 'file': #headers outside first line
            continue
        for param in csv_ints:
            if param in line:
                line[param] = int(float(line[param]))
        for param in csv_floats:
            if param in line:
                line[param] = float(line[param])
        csv_file.append(line)

    for i_exp, experiment in enumerate(experiments2):
        fig, ax = plt.subplots()
        current_rows = []
        for row in csv_file:
            isOk = True
            for res_name, res_val in experiment["restrictions"]:
                if isinstance(res_val, str):
                    if not bool(re.search(res_val, row[res_name])):
                        isOk = False
                        break
                else:
                    if not res_val == row[res_name]:
                        isOk = False
                        break
            if isOk:
                current_rows.append(row)

        if not current_rows:
            continue

        # print(experiment)
        # print(len(current_rows))
        # continue

        algos = set()
        for row in current_rows:
            if row[csv_algo] not in algos:
                algos.add(row[csv_algo])

        for i_algo, algo in enumerate(algos):
            algo_rows = filter(
                lambda row: True if row[csv_algo] == algo else False, current_rows)

            param_to_plot_by = experiment["x_param"]
            plot_by_values = set()

            for row in algo_rows:
                if param_to_plot_by in row:
                    val = row[param_to_plot_by]
                    if val not in plot_by_values:
                        plot_by_values.add(val)

            x = sorted(plot_by_values)
            print(x)

            # For each test there are different params (N, batch, grasp). On X axis will be param with most values
            print(param_to_plot_by)

            # For now we allow one additional parameter to plot (we produce a plot for each of its values)

            y = [''] * len(x)

            for row in algo_rows:
                for i_xval, xval in enumerate(x):
                    if row[param_to_plot_by] == xval:
                        y[i_xval] = row[csv_overall] / 1000
                        x[i_xval] = row[experiment["x_show"]]
                        if experiment["x_show"] == csv_filename:
                            x[i_xval] = os.path.basename(x[i_xval]).split('.')[0]


            ax.set_xlabel(param_to_plot_by)
            ax.set_ylabel("Time overall (s)")

            if experiment["type"] == "bars":
                width = 0.3
                real_x = np.arange(len(x))
                bar_shift = (i_algo + 0.5 - (len(algos) / 2.0)) * width
                ax.bar(real_x + bar_shift, y, width, label=algo)
                ax.set_xticklabels(x)
                ax.set_xticks(real_x)
            else:
                ax.plot(x, y, label=algo)

        suffix = ""
        for res_name, res_val in experiment["restrictions"]:
            suffix += "#" + res_name + ":" + str(res_val)

        ax.legend()
        if experiment["type"] == "bars":
            plt.xticks(rotation=30, ha='right')
            # fig.subplots_adjust(bottom=0.2)
            fig.tight_layout()
        else:
            ax.set_xscale("log")
            # fig.constrained_layout()
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        fig.savefig(suffix + ".png")
