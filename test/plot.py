# usePgf = True
usePgf = False

import matplotlib as mpl
if usePgf:
    mpl.use('pgf')
else:
    mpl.use('Agg')

import re
import io
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
csv_BFS = "BFS"
csv_find_bridges = "Find bridges"
csv_spanning_tree = "Spanning Tree"
csv_list_rank = "List rank"
csv_distance_parent = "Distance & parent"

csv_ints = []
csv_ints.extend([csv_N, csv_grasp, csv_batch])
csv_floats = []
csv_floats.extend([csv_overall, csv_BFS, csv_spanning_tree,
                   csv_list_rank, csv_distance_parent, csv_find_bridges])

csv_detailed_fields = [csv_BFS, csv_spanning_tree,
                       csv_list_rank, csv_distance_parent, csv_find_bridges]


# Here we declare what tests consist an experiment.
# An experiment is described by a regex that should be contained in input's filename
experiments = [
    # {"restrictions": [(csv_filename, "E1"), (csv_grasp, -1)],
    #  "x_param": csv_N, "x_show": csv_N, "type": "lines"},
    # {"restrictions": [(csv_filename, "E1"), (csv_grasp, 1000)],
    #  "x_param": csv_N, "x_show": csv_N, "type": "lines"},
    # {"restrictions": [(csv_filename, "E2")],
    #  "x_param": csv_batch, "x_show": csv_batch, "type": "lines"},
    # {"restrictions": [(csv_filename, "E3")],
    #  "x_param": csv_grasp, "x_show": csv_grasp, "type": "lines"},
    # {"restrictions": [(csv_filename, "kron")],
    #  "x_param": csv_N, "x_show": csv_N, "type": "lines"},
    # {"restrictions": [(csv_filename, "road-g|road-a|road-r|road-b|roadNet")],
    #  "x_param": csv_N, "x_show": csv_filename, "type": "bars"},
    # {"restrictions": [(csv_filename, "USA-road")],
    #  "x_param": csv_N, "x_show": csv_filename, "type": "bars"},
    # {"restrictions": [(csv_filename, "citationCiteseer|Patents|coPapersDBLP|delaunay_n24|in-2004|internet")],
    #  "x_param": csv_N, "x_show": csv_filename, "type": "bars"},
    # {"restrictions": [(csv_filename, "roadNet|osm")],
    #  "x_param": csv_N, "x_show": csv_filename, "type": "bars"},
    # {"restrictions": [(csv_filename, "com-LiveJournal|hollywood|Stanford")],
    #  "x_param": csv_N, "x_show": csv_filename, "type": "bars"},
    # {"restrictions": [(csv_filename, "soc|wikipedia2009|ca-hollywood-2009")],
    #  "x_param": csv_N, "x_show": csv_filename, "type": "bars"},
    # {"restrictions": [(csv_filename, "amazon0601|as-skitter|soc-LiveJournal1")],
    #  "x_param": csv_N, "x_show": csv_filename, "type": "bars"},
    {"restrictions": [(csv_filename, "kron_g500-logn21|USA-road-d.USA")],
     "x_param": csv_N, "x_show": csv_filename, "type": "detailed"},
]

csv_input = []
with open(sys.argv[1], 'r') as input:
    for line in input:
        if line.split(',')[0] == 'file':
            csv_input.append("")
        csv_input[-1] += line


csv_file = []
for input in csv_input:
    csv_reader = csv.DictReader(input.splitlines())
    for line in csv_reader:
        if line[csv_filename] == 'file':  # headers outside first line
            continue
        for param in csv_ints:
            if param in line:
                line[param] = int(float(line[param]))
        for param in csv_floats:
            if param in line:
                line[param] = float(line[param])
        csv_file.append(line)

for i_exp, experiment in enumerate(experiments):
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
        if experiment["type"] == "detailed":
            y = [[] for _ in range(len(x))]
        else:
            y = [''] * len(x)

        for row in algo_rows:
            for i_xval, xval in enumerate(x):
                if row[param_to_plot_by] == xval:
                    if experiment["type"] == "detailed":
                        for field in csv_detailed_fields:
                            if field in row:
                                y[i_xval].append(row[field])
                            else:
                                y[i_xval].append(0)
                    else:
                        y[i_xval] = row[csv_overall] / 1000.0
                    x[i_xval] = row[experiment["x_show"]]
                    if experiment["x_show"] == csv_filename:
                        x[i_xval] = os.path.basename(x[i_xval])

        print(y)

        ax.set_xlabel(param_to_plot_by)
        ax.set_ylabel("Time overall (s)")
        if experiment["type"] == "detailed":
            height = 0.3
            real_x = np.arange(len(x))
            y_sum = [sum(i) for i in y]
            bar_shift = (i_algo + 0.5 - (len(algos) / 2.0)) * height
            for i_field, field in reversed(list(enumerate(csv_detailed_fields))):
                if y[0][i_field] == 0:
                    alg_label = None
                else:
                    alg_label = algo + "-" + field
                ax.barh(real_x + bar_shift, y_sum, height, label=alg_label)

                for i_xval, xval in enumerate(x):
                    y_sum[i_xval] -= y[i_xval][i_field]
            ax.set_yticklabels(x)
            ax.set_yticks(real_x)
            # ax.barh(real_x + bar_shift, y, height, label=algo)
            # ax.barh(real_x + bar_shift, [i-0.2 for i in y], height, label=(algo + "aaa"))
        elif experiment["type"] == "bars":
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
        suffix += "_" + res_name + "-" + str(res_val)

    ax.legend()
    if experiment["type"] == "bars":
        plt.xticks(rotation=30, ha='right')
        # fig.subplots_adjust(bottom=0.2)
        fig.tight_layout()
        ax.set_yscale("log")
    elif experiment["type"] == "detailed":
        pass
    else:
        ax.set_xscale("log")
        ax.set_yscale("log")
        # fig.constrained_layout()
    ax.yaxis.set_major_formatter(ScalarFormatter())
    # plt.ylim(bottom=0)

    if usePgf:
        plt.rcParams.update(
            {'pgf.texsystem': 'pdflatex', 'font.family': 'serif', 'pgf.rcfonts': False, })
        plt.gcf().set_size_inches(w=3.34, h=2.6)
        plt.gcf().tight_layout()
        fig.savefig(suffix + ".pgf")
    else:
        fig.savefig(suffix + ".png")
