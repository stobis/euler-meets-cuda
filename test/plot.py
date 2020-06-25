usePgf = True
# usePgf = False

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
csv_M = 'M'
csv_Q = "# Q"
csv_algo = 'algo'
csv_overall = 'Overall'
csv_grasp = 'grasp'
csv_batch = 'batch'
csv_BFS = "BFS"
csv_find_bridges = "Find bridges"
csv_naive_bridges = "Naive bridges"
csv_spanning_tree = "Spanning Tree"
csv_list_rank = "List rank"
csv_distance_parent = "Distance and parent"
csv_preprocessing = "Preprocessing"
csv_queries = "Queries"

csv_ints = []
csv_ints.extend([csv_N, csv_M, csv_grasp, csv_batch])
csv_floats = []
csv_floats.extend([csv_overall, csv_BFS, csv_spanning_tree,
                   csv_list_rank, csv_distance_parent, csv_find_bridges, csv_naive_bridges, csv_preprocessing, csv_queries, csv_Q])

csv_detailed_fields = [csv_BFS, csv_spanning_tree,
                       csv_list_rank, csv_distance_parent, csv_find_bridges, csv_naive_bridges]

hatches = ['++', '**', 'oo', '----', '||', '..']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

sizes_in_inches = {"regular": (3.34, 2.6), "wide": (
    6.68, 2.6), "huge": (6.68, 7), "allGraphs": (10, 25)}
# sizes_in_inches = {"regular": (5, 3), "wide": (8, 3)}


# Here we declare what tests consist an experiment.
# An experiment is described by a regex that should be contained in input's filename
experiments = [
    {"restrictions": [(csv_filename, "E1"), (csv_grasp, 1000)],
     "x_param": csv_N, "x_show": csv_N, "type": "lines_xM", "size": sizes_in_inches["regular"],
     "y_param": csv_overall},
    {"restrictions": [(csv_filename, "E1"), (csv_grasp, -1)],
     "x_param": csv_N, "x_show": csv_N, "type": "lines_xM", "size": sizes_in_inches["regular"],
     "y_param": csv_overall},
    {"restrictions": [(csv_filename, "E1"), (csv_grasp, 1000)],
     "x_param": csv_N, "x_show": csv_N, "type": "lines_xM", "size": sizes_in_inches["regular"],
     "y_param": csv_preprocessing},
    {"restrictions": [(csv_filename, "E1"), (csv_grasp, -1)],
     "x_param": csv_N, "x_show": csv_N, "type": "lines_xM", "size": sizes_in_inches["regular"],
     "y_param": csv_preprocessing},
    {"restrictions": [(csv_filename, "E1"), (csv_grasp, 1000)],
     "x_param": csv_N, "x_show": csv_N, "type": "lines_xM", "size": sizes_in_inches["regular"],
     "y_param": csv_queries},
    {"restrictions": [(csv_filename, "E1"), (csv_grasp, -1)],
     "x_param": csv_N, "x_show": csv_N, "type": "lines_xM", "size": sizes_in_inches["regular"],
     "y_param": csv_queries},

    {"restrictions": [(csv_filename, "E1"), (csv_grasp, 1000)],
     "x_param": csv_N, "x_show": csv_N, "type": "lines_xM", "size": sizes_in_inches["regular"],
     "y_param": csv_N, "y_divide": csv_preprocessing},
    {"restrictions": [(csv_filename, "E1"), (csv_grasp, -1)],
     "x_param": csv_N, "x_show": csv_N, "type": "lines_xM", "size": sizes_in_inches["regular"],
     "y_param": csv_N, "y_divide": csv_preprocessing},
    {"restrictions": [(csv_filename, "E1"), (csv_grasp, 1000)],
     "x_param": csv_N, "x_show": csv_N, "type": "lines_xM", "size": sizes_in_inches["regular"],
     "y_param": csv_Q, "y_divide": csv_queries},
    {"restrictions": [(csv_filename, "E1"), (csv_grasp, -1)],
     "x_param": csv_N, "x_show": csv_N, "type": "lines_xM", "size": sizes_in_inches["regular"],
     "y_param": csv_Q, "y_divide": csv_queries},


    {"restrictions": [(csv_filename, "E2")],
     "x_param": csv_batch, "x_show": csv_batch, "type": "lines", "size": sizes_in_inches["regular"]},
    {"restrictions": [(csv_filename, "E2")],
     "x_param": csv_batch, "x_show": csv_batch, "type": "lines", "size": sizes_in_inches["regular"],
     "y_param": csv_Q, "y_divide": csv_queries},

    {"restrictions": [(csv_filename, "E3")],
     "x_param": csv_grasp, "x_show": csv_grasp, "type": "lines", "size": sizes_in_inches["regular"]},

    {"restrictions": [(csv_filename, "kron")],
     "x_param": csv_filename, "x_show": csv_N, "type": "lines", "size": sizes_in_inches["regular"]},
    {"restrictions": [(csv_filename, "road-a|road-g|road-d.USA|road-d.CTR|road-d.W|road-d.E")],
     "x_param": csv_filename, "x_show": csv_filename, "type": "bars", "size": sizes_in_inches["regular"]},
    {"restrictions": [(csv_filename, "cit-Patents|soc-Live|ca-hollywood|socfb-A-anon|wikipedia")],
     "x_param": csv_filename, "x_show": csv_filename, "type": "bars", "size": sizes_in_inches["regular"]},

    {"restrictions": [(csv_filename, "cit-Patents|soc-Live|ca-hollywood|socfb-A-anon|wikipedia|road-a|road-g|road-d.USA|road-d.CTR|road-d.W|road-d.E")],
     "x_param": csv_filename, "x_show": csv_filename, "type": "bars", "size": sizes_in_inches["wide"]},
    # {"restrictions": [(csv_filename, "road-g|road-a|road-r|road-b|roadNet")],
    #  "x_param": csv_N, "x_show": csv_filename, "type": "bars", "size": sizes_in_inches["regular"]},
    # {"restrictions": [(csv_filename, "USA-road")],
    #  "x_param": csv_N, "x_show": csv_filename, "type": "bars", "size": sizes_in_inches["regular"]},
    # {"restrictions": [(csv_filename, "citationCiteseer|Patents|coPapersDBLP|delaunay_n24|in-2004|internet")],
    #  "x_param": csv_N, "x_show": csv_filename, "type": "bars", "size": sizes_in_inches["regular"]},
    # {"restrictions": [(csv_filename, "roadNet|osm")],
    #  "x_param": csv_N, "x_show": csv_filename, "type": "bars", "size": sizes_in_inches["regular"]},
    # {"restrictions": [(csv_filename, "com-LiveJournal|hollywood|Stanford")],
    #  "x_param": csv_N, "x_show": csv_filename, "type": "bars", "size": sizes_in_inches["regular"]},
    # {"restrictions": [(csv_filename, "soc|wikipedia2009|ca-hollywood-2009")],
    #  "x_param": csv_N, "x_show": csv_filename, "type": "bars", "size": sizes_in_inches["regular"]},
    # {"restrictions": [(csv_filename, "amazon0601|as-skitter|soc-LiveJournal1")],
    #  "x_param": csv_N, "x_show": csv_filename, "type": "bars", "size": sizes_in_inches["regular"]},
    {"restrictions": [(csv_filename, "kron_g500-logn21|USA-road-d.CTR|cit-Patents")],
     "x_param": csv_filename, "x_show": csv_filename, "type": "detailed", "size": sizes_in_inches["wide"]},

    # {"restrictions": [(csv_filename, "road-g|road-d.USA|road-d.CTR|road-d.W|road-d.E")],
    #  "x_param": csv_filename, "x_show": csv_filename, "type": "detailed", "size": sizes_in_inches["huge"]},
    # {"restrictions": [(csv_filename, "cit-Patents|soc-Live|ca-hollywood|socfb-A-anon|wikipedia")],
    #  "x_param": csv_filename, "x_show": csv_filename, "type": "detailed", "size": sizes_in_inches["huge"]}
    {"restrictions": [(csv_filename, "cit-Patents|soc-Live|ca-hollywood|socfb-A-anon|wikipedia|road-g|road-d.USA|road-d.CTR|road-d.W|road-d.E")],
     "x_param": csv_filename, "x_show": csv_filename, "type": "detailed", "size": sizes_in_inches["huge"]},
    {"restrictions": [(csv_filename, "")],
     "x_param": csv_M, "x_show": csv_filename, "type": "detailed", "size": sizes_in_inches["allGraphs"]},
]

filename_order = [
    "kron_g500-logn16",
    "kron_g500-logn17",
    "kron_g500-logn18",
    "kron_g500-logn19",
    "kron_g500-logn20",
    "kron_g500-logn21",
    "web-wikipedia2009",
    "cit-Patents",
    "socfb-A-anon",
    "soc-LiveJournal1",
    "ca-hollywood-2009",
    "USA-road-d.E",
    "USA-road-d.W",
    "great-britain-osm",
    "asia-osm",
    "USA-road-d.CTR",
    "USA-road-d.USA"]

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
    if "y_param" not in experiment:
        experiment["y_param"] = csv_overall

    fig, ax = plt.subplots()
    xSize, ySize = experiment["size"]
    fig.set_size_inches(xSize, ySize, forward=True)
    font_size = 8
    plt.rcParams.update({'font.size': font_size})
    # plt.rc('font', size=font_size)
    # plt.rc('axes', titlesize=font_size)     # fontsize of the axes title
    # plt.rc('axes', labelsize=font_size)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=font_size)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=font_size)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=font_size)    # legend fontsize
    # plt.rc('figure', titlesize=font_size)
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

    algos = set()
    for row in current_rows:
        if row[csv_algo] not in algos:
            algos.add(row[csv_algo])

    if "tarjan" in algos:
        algos = ["naive", "tarjan", "hybrid"]

    for i_algo, algo in reversed(list(enumerate(algos))):
        algo_rows = filter(
            lambda row: True if row[csv_algo] == algo else False, current_rows)

        param_to_plot_by = experiment["x_param"]
        plot_by_values = set()

        for row in algo_rows:
            if param_to_plot_by in row:
                val = row[param_to_plot_by]
                if val not in plot_by_values:
                    plot_by_values.add(val)

        if param_to_plot_by == csv_filename:
            positions = {}
            for var in plot_by_values:
                for i_fn, fn in enumerate(filename_order):
                    if fn in var:
                        positions[var] = i_fn
                        break
            x = sorted(plot_by_values, key=lambda val: positions[val])
        else:
            x = sorted(plot_by_values)

        print(x)

        if experiment["type"] == "detailed":
            y = [[] for _ in range(len(x))]
        else:
            y = [''] * len(x)

        # Prepare data
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
                        y[i_xval] = row[experiment["y_param"]]
                        if "y_divide" in experiment:
                            y[i_xval] /= row[experiment["y_divide"]] * 1000
                    x[i_xval] = row[experiment["x_show"]]
                    if experiment["x_show"] == csv_filename:
                        x[i_xval] = os.path.basename(x[i_xval])
                        x[i_xval] = x[i_xval].replace(".gr.bin", "")
                        x[i_xval] = x[i_xval].replace(".mtx.bin", "")
                        x[i_xval] = x[i_xval].replace(".txt.bin", "")

        # Plot data
        if experiment["type"] == "detailed":
            # ax.set_ylabel(experiment["x_show"])
            ax.set_xlabel("Time overall (ms)")
            height = 0.3
            real_x = np.arange(len(x))
            y_sum = [sum(i) for i in y]
            bar_shift = (i_algo + 0.5 - (len(algos) / 2.0)) * height
            for i_field, field in reversed(list(enumerate(csv_detailed_fields))):
                if y[0][i_field] == 0:
                    alg_label = None
                else:
                    alg_label = algo + "-" + field
                ax.barh(real_x + bar_shift, y_sum, height, label=alg_label,
                        color='white', edgecolor=colors[i_algo], hatch=hatches[i_field])

                for i_xval, xval in enumerate(x):
                    y_sum[i_xval] -= y[i_xval][i_field]

            ax.set_yticks(real_x)
            ax.set_yticklabels(x)
            ax.invert_yaxis()
        elif experiment["type"] == "bars":
            # ax.set_xlabel(experiment["x_show"])
            ax.set_ylabel("Time " + experiment["y_param"] + " (ms)")
            width = 0.3
            real_x = np.arange(len(x))
            bar_shift = (i_algo + 0.5 - (len(algos) / 2.0)) * width
            ax.bar(real_x + bar_shift, y, width,
                   label=algo, color=colors[i_algo], bottom=10)
            ax.set_xticklabels(x)
            ax.set_xticks(real_x)
        elif experiment["type"] == "lines_xM":
            ax.set_xlabel(experiment["x_show"], fontsize=font_size)
            ax.set_ylabel(
                "Time " + experiment["y_param"] + " (ms)", fontsize=font_size)
            if "y_divide" in experiment:
                ax.set_ylabel(
                    "Million " + experiment["y_divide"] + "/s", fontsize=font_size)
            real_x = np.arange(len(x))
            string_x = [str(val/1000000) + "M" for val in x]
            ax.plot(real_x, y, label=algo, color=colors[i_algo], marker=".")
            if "y_divide" in experiment:
                ax.set_ylim(0.1, 1000)
            ax.set_xticklabels(string_x)
            ax.set_xticks(real_x)
        else:
            ax.set_xlabel(experiment["x_show"], fontsize=font_size)
            ax.set_ylabel(
                "Time " + experiment["y_param"] + " (ms)", fontsize=font_size)
            if "y_divide" in experiment:
                ax.set_ylabel(
                    "Million " + experiment["y_divide"] + "/s", fontsize=font_size)
            ax.plot(x, y, label=algo, color=colors[i_algo], marker=".")

    suffix = ""
    for res_name, res_val in experiment["restrictions"]:
        suffix += "_" + res_name + "-" + str(res_val)

    suffix += "x=" + experiment["x_param"]
    suffix += "y=" + experiment["y_param"]
    if "y_divide" in experiment:
        suffix += "%" + experiment["y_divide"]
    suffix += experiment["type"]

    # Format plot
    if experiment["type"] == "bars":
        plt.xticks(rotation=30, ha='right')
        # fig.subplots_adjust(bottom=0.2)
        fig.tight_layout()
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])
    elif experiment["type"] == "detailed":
        if experiment["size"] == sizes_in_inches["huge"]:
            fig.tight_layout(rect=[0, 0.1, 1, 1])
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(handles[::-1], labels[::-1], loc='center', bbox_to_anchor=(0.4, -0.15),
                               fancybox=True, shadow=True, ncol=3)
        elif experiment["size"] == sizes_in_inches["allGraphs"]:
            fig.tight_layout(rect=[0, 0.05, 1, 1])
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(handles[::-1], labels[::-1], loc='center', bbox_to_anchor=(0.4, -0.05),
                               fancybox=True, shadow=True, ncol=3)
        else:
            fig.tight_layout(rect=[0, 0.2, 1, 1])
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(handles[::-1], labels[::-1], loc='center', bbox_to_anchor=(0.4, -0.45),
                               fancybox=True, shadow=True, ncol=3)
    elif experiment["type"] == "lines_xM":
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        fig.tight_layout()
        # ax.set_xscale("log")
        ax.set_yscale("log")
        if "y_divide" in experiment:
            ax.yaxis.set_major_formatter(ScalarFormatter())
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])
        # fig.constrained_layout()
    else:  # lines
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        fig.tight_layout()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])
        # fig.constrained_layout()
    # plt.ylim(bottom=0)

    # Save
    if usePgf:
        plt.rcParams.update(
            {'pgf.texsystem': 'pdflatex', 'font.family': 'serif', 'pgf.rcfonts': False, })
        plt.gcf().set_size_inches(w=xSize, h=ySize)
        if experiment["type"] == "detailed":
            fig.savefig(suffix + ".pgf", bbox_extra_artists=(legend))
        else:
            fig.savefig(suffix + ".pgf")
    else:
        # fig.set_size_inches(10,5, forward=True)
        plt.rcParams.update(
            {'pgf.texsystem': 'pdflatex', 'font.family': 'serif', 'pgf.rcfonts': False, })
        plt.gcf().set_size_inches(w=xSize, h=ySize)
        if experiment["type"] == "detailed":
            fig.savefig(suffix + ".png", bbox_extra_artists=(legend))
        else:
            fig.savefig(suffix + ".png")
