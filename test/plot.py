# coding: utf-8

usePgf = True
# usePgf = False

import matplotlib as mpl
if usePgf:
    mpl.use('pgf')
else:
    mpl.use('Agg')


from cycler import cycler
import re
import io
import os
import sys
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import numpy as np

plt.rcParams.update({'pgf.texsystem': 'pdflatex', 'font.family': 'serif', 'pgf.rcfonts': False, })

csv_filename = "file"
csv_N = 'N'
csv_M = 'M'
csv_Q = "numQ"
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
csv_avg_height = "AvgHeight"
csv_max_height = "MaxHeight"
csv_marking = "Marking"

csv_ints = []
csv_ints.extend([csv_N, csv_M, csv_Q, csv_grasp, csv_batch, csv_max_height])
csv_floats = []
csv_floats.extend([csv_overall, csv_BFS, csv_spanning_tree,
                   csv_list_rank, csv_distance_parent, csv_find_bridges, csv_naive_bridges, csv_preprocessing, csv_queries, csv_avg_height, csv_marking])

csv_detailed_fields = [csv_BFS, csv_spanning_tree,
                       csv_list_rank, csv_distance_parent, csv_find_bridges, csv_naive_bridges, csv_marking]

hatches = ['++', '**', 'oo', '----', '||', '..']
field_colors = {"tarjan": {"Spanning Tree": "#0f2231", "List rank": "#2c6593", "Find bridges": "#6ca5d3"},
                "naive": {"BFS": "#cc6600", "Naive bridges": "#ff9933"},
                "hybrid": { "Spanning Tree": "#1a1a1a", "List rank": "#4d4d4d", "Distance and parent": "#808080", "Naive bridges": "#b3b3b3"},
                "multicore": {"BFS": "#961b1c", "Marking": "#de5253"} }
algo_colors = {"CUDA Inlabel": "#377eb8", "CUDA Naive": "#ff7f00", "CPU Inlabel": "#4daf4a", "Multicore CPU Inlabel": "#d62728",
               "tarjan": "#377eb8", "naive": "#ff7f00", "hybrid": "#999999", "cpu": "#4daf4a",
               "multicore": "#d62728"}
algo_markers = {"CUDA Inlabel": ".", "CUDA Naive": "x", "CPU Inlabel": "*", "Multicore CPU Inlabel": "|",
               "tarjan": ".", "naive": "x", "hybrid": "*", "cpu": "*", "multicore": "|"}

sizes_in_inches = {"regular": (3.465, 2.55), "wide": (
    7.1, 2.4), "huge": (7.1, 8), "allGraphs": (10, 25), "wideDetailed":(7.1 , 3)}
# sizes_in_inches = {"regular": (5, 3), "wide": (8, 3)}

E1_div_ylim = (100000,1000000000)
E3_ylim = (0.1, 60)
E4_ylim = (0.01, 100)
E4_ylim_div = (100000,1000000000)
bridges_bars_ylim = (0.04, 10)
bridges_kron_ylim = (0.01, 5.5)

func_formatter = ticker.FuncFormatter(lambda y, _: '{:g}'.format(y))

#labels for plot axes
label_lca_N = "Number of nodes"
label_lca_preprocessing_over_s = "Nodes preprocessed per second"
label_lca_queries_over_s = "Queries answered per second"
label_lca_queries_over_N = "Queries-to-nodes ratio"
label_lca_avg_depth = "Average node depth"
label_lca_batch_size = "Batch size"
label_time_overall_ms = "Total time [ms]"
label_time_overall_s = "Total time [s]"

#labels for algorithms
algo_labels = {"CUDA Inlabel": "GPU Inlabel", "CUDA Naive": u"GPU Naïve", "CPU Inlabel": "Single-core CPU Inlabel", "Multicore CPU Inlabel": "Multi-core CPU Inlabel",
               "tarjan": "GPU TV", "naive": "GPU CK", "cpu": "Single-core CPU DFS", "hybrid": "GPU Hybrid", "multicore": "Multi-core CPU CK"}

algo_field_labels = {"Find bridges" : "Detect bridges", "Naive bridges": "Mark non-bridges",
                     "Spanning Tree" : "Spanning tree", "List rank" : "Euler tour",
                     "Distance and parent" : "Levels and parents", "BFS" : "BFS", "Marking": "Marking"}

# Here we declare what tests consist an experiment.
experiments = [
    #LCA
    {"restrictions": [(csv_filename, "E1"), (csv_grasp, 1000)],
     "x_param": csv_N, "x_show": csv_N, "type": "lines_xM", "size": sizes_in_inches["regular"],
     "y_param": csv_N, "y_divide": csv_preprocessing, "ylim": E1_div_ylim, 
     "x_label": label_lca_N, "y_label": label_lca_preprocessing_over_s, "ncol": 2},
    {"restrictions": [(csv_filename, "E1"), (csv_grasp, -1)],
     "x_param": csv_N, "x_show": csv_N, "type": "lines_xM", "size": sizes_in_inches["regular"],
     "y_param": csv_N, "y_divide": csv_preprocessing, "ylim": E1_div_ylim,
     "x_label": label_lca_N, "y_label": label_lca_preprocessing_over_s, "ncol": 2},
    {"restrictions": [(csv_filename, "E1"), (csv_grasp, 1000)],
     "x_param": csv_N, "x_show": csv_N, "type": "lines_xM", "size": sizes_in_inches["regular"],
     "y_param": csv_Q, "y_divide": csv_queries, "ylim": E1_div_ylim,
     "x_label": label_lca_N, "y_label": label_lca_queries_over_s, "ncol": 1},
    {"restrictions": [(csv_filename, "E1"), (csv_grasp, -1)],
     "x_param": csv_N, "x_show": csv_N, "type": "lines_xM", "size": sizes_in_inches["regular"],
     "y_param": csv_Q, "y_divide": csv_queries, "ylim": E1_div_ylim,
     "x_label": label_lca_N, "y_label": label_lca_queries_over_s, "ncol": 2},


    # {"restrictions": [(csv_filename, "E2")],
    #  "x_param": csv_batch, "x_show": csv_batch, "type": "lines", "size": sizes_in_inches["regular"],
    #  "yformatter": func_formatter},
    {"restrictions": [(csv_filename, "E2")],
     "x_param": csv_batch, "x_show": csv_batch, "type": "lines", "size": sizes_in_inches["regular"],
     "y_param": csv_Q, "y_divide": csv_queries, "ylim": E1_div_ylim,
     "x_label": label_lca_batch_size, "y_label": label_lca_queries_over_s},

    {"restrictions": [(csv_filename, "E3")],
     "x_param": csv_avg_height, "x_show": csv_avg_height, "type": "lines", "size": sizes_in_inches["regular"],
     "yformatter": func_formatter, "ylim": E3_ylim,
     "x_label": label_lca_avg_depth, "y_label": label_time_overall_s},
    # {"restrictions": [(csv_filename, "E3")],
    #  "x_param": csv_max_height, "x_show": csv_max_height, "type": "lines", "size": sizes_in_inches["regular"],
    #  "yformatter": func_formatter, "ylim": E3_ylim}

    # {"restrictions": [(csv_filename, "E4"), (csv_grasp, 1000)],
    #  "x_param": csv_Q, "x_show": csv_Q, "type": "lines_xM", "size": sizes_in_inches["regular"],
    #  "y_param": csv_overall, "ylim": E4_ylim, "yformatter": func_formatter},
    {"restrictions": [(csv_filename, "E4"), (csv_grasp, -1)],
     "x_param": csv_Q, "x_show": csv_Q, "x_div": csv_N, "type": "lines_xM", "size": sizes_in_inches["regular"],
     "y_param": csv_overall, "ylim": E4_ylim, "yformatter": func_formatter,
     "x_label": label_lca_queries_over_N, "y_label": label_time_overall_s},

    # {"restrictions": [(csv_filename, "E4"), (csv_grasp, 1000)],
    #  "x_param": csv_Q, "x_show": csv_Q, "type": "lines_xM", "size": sizes_in_inches["regular"],
    #  "y_param": csv_Q, "y_divide": csv_queries, "ylim": E4_ylim_div},
    # {"restrictions": [(csv_filename, "E4"), (csv_grasp, -1)],
    #  "x_param": csv_Q, "x_show": csv_Q, "type": "lines_xM", "size": sizes_in_inches["regular"],
    #  "y_param": csv_Q, "y_divide": csv_queries, "ylim": E4_ylim_div},


    {"restrictions": [(csv_filename, "E5")],
     "x_param": csv_N, "x_show": csv_N, "type": "lines_xM", "size": sizes_in_inches["regular"],
     "y_param": csv_N, "y_divide": csv_preprocessing, "ylim": E1_div_ylim, 
     "x_label": label_lca_N, "y_label": label_lca_preprocessing_over_s, "ncol": 2},
    {"restrictions": [(csv_filename, "E5")],
     "x_param": csv_N, "x_show": csv_N, "type": "lines_xM", "size": sizes_in_inches["regular"],
     "y_param": csv_Q, "y_divide": csv_queries, "ylim": E1_div_ylim,
     "x_label": label_lca_N, "y_label": label_lca_queries_over_s, "ncol": 2},

    #Bridges
    {"restrictions": [(csv_filename, "kron")],
     "x_param": csv_filename, "x_show": csv_N, "type": "lines", "size": sizes_in_inches["regular"],
     "yformatter": func_formatter, "skipHybrid": True, "y_label" : label_time_overall_s,
     "x_label": label_lca_N, "ylim": bridges_kron_ylim},
    
    {"restrictions": [(csv_filename, "cit-Patents|soc-Live|ca-hollywood|socfb-A-anon|wikipedia|road-a|road-g|road-d.USA|road-d.CTR|road-d.W|road-d.E")],
     "x_param": csv_filename, "x_show": csv_filename, "type": "bars", "size": sizes_in_inches["wide"],
     "yformatter": func_formatter, "skipHybrid": True, "y_label": label_time_overall_s},

    {"restrictions": [(csv_filename, "kron_g500-logn21|soc-LiveJ|cit-Patents|road-great-britain|USA-road-d.CTR")],
     "x_param": csv_filename, "x_show": csv_filename, "type": "detailed", "size": sizes_in_inches["wideDetailed"],
     "skipHybrid": True, "x_label": label_time_overall_ms},

    {"restrictions": [(csv_filename, "kron_g500-logn19|kron_g500-logn2|cit-Patents|soc-Live|ca-hollywood|socfb-A-anon|wikipedia|road-g|road-d.USA|road-d.CTR|road-d.W|road-d.E")],
     "x_param": csv_filename, "x_show": csv_filename, "type": "detailed", "size": sizes_in_inches["huge"],
      "x_label": label_time_overall_ms},

    # {"restrictions": [(csv_filename, ".bin")],
    #  "x_param": csv_M, "x_show": csv_filename, "type": "detailed", "size": sizes_in_inches["allGraphs"]},
]

# Order of filenames on X in bridges
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

bridges_algo_order = ["cpu", "multicore", "naive", "tarjan", "hybrid"]
lca_algo_order = ["CPU Inlabel", "Multicore CPU Inlabel", "CUDA Naive","CUDA Inlabel"  ]

# TODO(Adrian python3?)
if len(sys.argv) < 2:
    print("Usage:\n\
    \t./python2 plot.py csv_to_parse.csv [dir_to_save_plots]\n")
    exit()

# Read input and do preprocessing
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
            if param in line and line[param]:
                line[param] = int(float(line[param]))
        for param in csv_floats:
            if param in line and line[param]:
                line[param] = float(line[param])
        csv_file.append(line)

for i_exp, experiment in enumerate(experiments):
    if "y_param" not in experiment: # by default we plot overall time
        experiment["y_param"] = csv_overall


    fig, ax = plt.subplots()
    xSize, ySize = experiment["size"]
    fig.set_size_inches(xSize, ySize, forward=True)
    font_size = 8
    plt.rcParams.update({'font.size': font_size})

    current_rows = []
    #Filter data we should plot in current image
    for row in csv_file:
        isOk = True
        for res_name, res_val in experiment["restrictions"]: #process restrictions to plot only what we want
            if isinstance(res_val, str): # for a string, restriction should be a substring of a field in a row
                if not bool(re.search(res_val, row[res_name])):
                    isOk = False
                    break
            else: # for a non-string restriction should equal a field in a row
                if not res_val == row[res_name]:
                    isOk = False
                    break
        if isOk:
            current_rows.append(row)

    if not current_rows:
        continue

    # Scan for algos to plot
    algos = set()
    for row in current_rows:
        if row[csv_algo] not in algos:
            algos.add(row[csv_algo])

    if "tarjan" in algos:
        algos = bridges_algo_order
    else:
        algos = lca_algo_order

    for i_algo, algo in reversed(list(enumerate(algos))):

        if algo == "hybrid" and "skipHybrid" in experiment and experiment["skipHybrid"]:
            if experiment["type"] == "detailed":
                ax.invert_yaxis()
            continue
        # Filter data of a given algorithm
        algo_rows = filter(
            lambda row: True if row[csv_algo] == algo else False, current_rows)
        
        if not algo_rows:
            continue

        plot_by_values = set()

        # Get all valid x-values
        for row in algo_rows:
            if experiment["x_param"] in row:
                val = row[experiment["x_param"]]
                if val not in plot_by_values:
                    plot_by_values.add(val)

        # If our x-value is a filename, sort by order from config
        if experiment["x_param"] == csv_filename:
            positions = {}
            for var in plot_by_values:
                for i_fn, fn in enumerate(filename_order):
                    if fn in var:
                        positions[var] = i_fn
                        break
            x = sorted(plot_by_values, key=lambda val: positions[val])
        else: 
            x = sorted(plot_by_values)

        # print(x)

        if experiment["type"] == "detailed":
            y = [[] for _ in range(len(x))]
        else:
            y = [''] * len(x)

        # Prepare data values for current plot
        for row in algo_rows:
            for i_xval, xval in enumerate(x):
                if row[experiment["x_param"]] == xval:
                    if experiment["type"] == "detailed":
                        for field in csv_detailed_fields:
                            if field in row:
                                y[i_xval].append(row[field])
                            else:
                                y[i_xval].append(0) # if our field is not in a current algo, use value of 0 which we will skip in plotting
                    else:
                        y[i_xval] = row[experiment["y_param"]]
                        if "y_divide" in experiment:
                            y[i_xval] /= row[experiment["y_divide"]] / 1000.0
                        elif experiment["type"] != "detailed": 
                            y[i_xval] /= 1000
                    x[i_xval] = row[experiment["x_show"]]
                    if experiment["x_show"] == csv_filename:
                        x[i_xval] = os.path.basename(x[i_xval])
                        x[i_xval] = x[i_xval].replace(".gr.bin", "")
                        x[i_xval] = x[i_xval].replace(".mtx.bin", "")
                        x[i_xval] = x[i_xval].replace(".txt.bin", "")

        # Plot data
        if experiment["type"] == "detailed":
            if algo == "cpu" or algo == "multicore":
                continue
            # ax.set_ylabel(experiment["x_show"])
            ax.set_xlabel("Time overall (ms)")
            if "skipHybrid" in experiment and experiment["skipHybrid"]:
                height = 0.4
                # height = 0.3
            else:
                height = 0.28
                # height = 0.20
            apparent_x = np.arange(len(x))
            y_sum = [sum(i) for i in y]
            if "skipHybrid" in experiment and experiment["skipHybrid"]:
                bar_shift = (i_algo -0.5 - ((len(algos) -1) / 2.0)) * height
            else:
                bar_shift = (i_algo -0.5 - ((len(algos) -0) / 2.0)) * height

            ax.invert_yaxis()
            for i_field, field in reversed(list(enumerate(csv_detailed_fields))):
                if y[0][i_field] == 0:
                    alg_label = None
                else:
                    alg_label = algo_labels[algo] + " " + algo_field_labels[field]
                    # ax.barh(apparent_x + bar_shift, y_sum, height, label=alg_label,
                        # color='white', edgecolor=algo_colors[algo], hatch=hatches[i_field], zorder=3)
                    ax.barh(apparent_x + bar_shift, y_sum, height, label=alg_label,
                        color=field_colors[algo][field], zorder=3)

                for i_xval, xval in enumerate(x):
                    y_sum[i_xval] -= y[i_xval][i_field]

            ax.set_yticks(apparent_x)
            ax.set_yticklabels(x)
        elif experiment["type"] == "bars":
            # ax.set_xlabel(experiment["x_show"])
            ax.set_ylabel("Time " + experiment["y_param"] + " (s)")
            # width = 0.3
            width = 0.2
            apparent_x = np.arange(len(x))
            ax.set_ylim(bridges_bars_ylim[0], bridges_bars_ylim[1])
            bar_shift = (i_algo + 0.5 - ((len(algos) - 1) / 2.0)) * width
            ax.bar(apparent_x + bar_shift, y, width,
                   label=algo_labels[algo], color=algo_colors[algo], zorder=1.99, bottom=0.02)
            ax.set_xticklabels(x)
            ax.set_xticks(apparent_x)
        elif experiment["type"] == "lines_xM":
            ax.set_ylabel(
                "Time " + experiment["y_param"] + " (s)", fontsize=font_size)
            if "y_divide" in experiment:
                ax.set_ylabel(
                    experiment["y_divide"] + "/s", fontsize=font_size)
            apparent_x = np.arange(len(x))
            if "x_div" in experiment:
                string_x = [val/8000000.0 for val in x]
                ax.set_xlabel("Queries/N")
            else:
                string_x = [str(val/1000000) + "M" for val in x]
                ax.set_xlabel(experiment["x_show"], fontsize=font_size)
            ax.plot(apparent_x, y, label=algo_labels[algo], color=algo_colors[algo], marker=algo_markers[algo])
            # ax.yaxis.grid(which="minor", linestyle='--', linewidth=.3)
            ax.set_xticklabels(string_x)
            ax.set_xticks(apparent_x)
        else:
            ax.set_xlabel(experiment["x_show"], fontsize=font_size)
            ax.set_ylabel(
                "Time " + experiment["y_param"] + " (s)", fontsize=font_size)
            if "y_divide" in experiment:
                ax.set_ylabel(
                    experiment["y_divide"] + "/s", fontsize=font_size)
            ax.plot(x, y, label=algo_labels[algo], color=algo_colors[algo], marker=algo_markers[algo])
            ax.set_xticks(x)
        
        if "x_label" in experiment:
            ax.set_xlabel(experiment["x_label"])
        if "y_label" in experiment:
            ax.set_ylabel(experiment["y_label"])
        


    # Format plot
    if experiment["type"] == "bars":
        plt.xticks(rotation=30, ha='right')
        # fig.subplots_adjust(bottom=0.2)
        fig.tight_layout()
        ax.set_yscale("log")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], ncol=4)
    elif experiment["type"] == "detailed":
        if experiment["size"] == sizes_in_inches["huge"]:
            fig.tight_layout(rect=[0, 0.08, 1, 1])

            rect = patches.Rectangle((-240,13.96),1342,1.2,linewidth=1,edgecolor='black',facecolor='white', clip_on=False)
            shadow = patches.Rectangle((-235,14.00),1342,1.2,linewidth=1,edgecolor='grey',facecolor='grey', clip_on=False)
            ax.add_patch(shadow)
            ax.add_patch(rect)

            handles_and_labels = zip(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1])
            hl_ck = [hl for hl in handles_and_labels if "CK" in hl[1]]
            handles_ck, labels_ck = zip(*hl_ck)

            hl_tv = [hl for hl in handles_and_labels if "TV" in hl[1]]
            handles_tv, labels_tv = zip(*hl_tv)

            hl_hy = [hl for hl in handles_and_labels if "Hybrid" in hl[1]]
            handles_hy, labels_hy = zip(*hl_hy)

            r = mpl.patches.Rectangle((0,0), 1, 1.5, fill=False, edgecolor='none', visible=False)

            labels_ck = [l[7::] for l in labels_ck] + ["GPU CK\hspace{13.15pt}"]
            handles_ck = handles_ck + (r,)

            labels_tv = [l[7::] for l in labels_tv] + ["GPU TV\hspace{13.3pt}"]
            handles_tv = handles_tv + (r,)

            labels_hy = [l[11::] for l in labels_hy] + ["GPU Hybrid"]
            handles_hy = handles_hy + (r,)

            legend = ax.legend(handles_ck[::-1], labels_ck[::-1], loc='lower left', bbox_to_anchor=(-.27,-0.104),
                               fancybox=True, shadow=False, ncol=3, frameon=False)
            legend1 = ax.legend(handles_tv[::-1], labels_tv[::-1], loc='lower left', bbox_to_anchor=(-.27,-0.129),
                               fancybox=True, shadow=False, ncol=4, frameon=False)
            legend2 = ax.legend(handles_hy[::-1], labels_hy[::-1], loc='lower left', bbox_to_anchor=(-.27, -.154),
                               fancybox=True, shadow=False, ncol=5, frameon=False)
            plt.gca().add_artist(legend)
            plt.gca().add_artist(legend1)


        elif experiment["size"] == sizes_in_inches["allGraphs"]:
            fig.tight_layout(rect=[0, 0.05, 1, 1])
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(handles[::-1], labels[::-1], loc='center', bbox_to_anchor=(0.4, -0.05),
                               fancybox=True, shadow=True, ncol=3)
        elif experiment["size"] == sizes_in_inches["wideDetailed"]:
            fig.tight_layout(rect=[0, 0.13, 1, 1])

            rect = patches.Rectangle((0,5.75),750,1,linewidth=1,edgecolor='black',facecolor='white', clip_on=False)
            shadow = patches.Rectangle((5,5.8),750,1,linewidth=1,edgecolor='grey',facecolor='grey', clip_on=False)
            ax.add_patch(shadow)
            ax.add_patch(rect)

            handles_and_labels = zip(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1])

            hl_ck = [hl for hl in handles_and_labels if "CK" in hl[1]]
            handles_ck, labels_ck = zip(*hl_ck)

            hl_tv = [hl for hl in handles_and_labels if "TV" in hl[1]]
            handles_tv, labels_tv = zip(*hl_tv)

            r = mpl.patches.Rectangle((0,0), 1, 1.5, fill=False, edgecolor='none', visible=False)

            labels_ck = [l[7::] for l in labels_ck] + ["GPU CK"]
            handles_ck = handles_ck + (r,)

            labels_tv = [l[7::] for l in labels_tv] + ["GPU TV"]
            handles_tv = handles_tv + (r,)

            legend = ax.legend(handles_ck[::-1], labels_ck[::-1], loc='lower left', bbox_to_anchor=(-.05,-0.35),
                               fancybox=True, shadow=False, ncol=3, frameon=False)
            legend1 = ax.legend(handles_tv[::-1], labels_tv[::-1], loc='lower left', bbox_to_anchor=(-.05,-0.43),
                               fancybox=True, shadow=False, ncol=4, frameon=False)
            plt.gca().add_artist(legend)
        else:
            fig.tight_layout(rect=[0, 0.2, 1, 1])
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(handles[::-1], labels[::-1], loc='center', bbox_to_anchor=(0.4, -0.45),
                               fancybox=True, shadow=True, ncol=2)
    else:  # lines
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        if experiment["type"] == "lines":
            ax.set_xscale("log")
        ax.set_yscale("log")
        # ax.yaxis.set_major_formatter(ScalarFormatter())
        handles, labels = ax.get_legend_handles_labels()
        if "ncol" in experiment:
            ax.legend(handles[::-1], labels[::-1], ncol=experiment["ncol"], columnspacing=0.6)
        else:
            ax.legend(handles[::-1], labels[::-1], ncol=1)
            
        fig.tight_layout()

    if "yformatter" in experiment:
        ax.yaxis.set_major_formatter(experiment["yformatter"])
    if "ylim" in experiment:
        ax.set_ylim(experiment["ylim"])
    
    if experiment["type"] != "detailed":
        ax.yaxis.grid(b=True, which="major", zorder=0)
    else:
        ax.xaxis.grid(b=True, which="major", zorder=0)
    # ax.grid(True)


    # Saving path
    if len(sys.argv) <= 2:
        dir_to_save_to = os.path.join("testResults", "plots")
    else:
        dir_to_save_to = sys.argv[2]

    if not os.path.exists(dir_to_save_to):
        os.makedirs(dir_to_save_to)

    # Saving filename
    filename_to_save = ""
    for res_name, res_val in experiment["restrictions"]:
        filename_to_save += "_" + res_name + "-" + str(res_val)

    filename_to_save += "x=" + experiment["x_param"]
    filename_to_save += "y=" + experiment["y_param"]
    if "y_divide" in experiment:
        filename_to_save += "div" + experiment["y_divide"]

    filename_to_save += experiment["type"]
    filename_to_save = filename_to_save.replace("# ", "num")


    # Save
    plt.gcf().set_size_inches(w=xSize, h=ySize)

    if usePgf:
        if experiment["type"] == "detailed":
            fig.savefig(os.path.join(dir_to_save_to, filename_to_save + ".pgf"),
                        bbox_extra_artists=(legend))
        else:
            fig.savefig(os.path.join(dir_to_save_to, filename_to_save + ".pgf"))
    else:
        # fig.set_size_inches(10,5, forward=True)
        if experiment["type"] == "detailed":
            fig.savefig(os.path.join(dir_to_save_to, filename_to_save + ".png"),
                        bbox_extra_artists=(legend))
        else:
            fig.savefig(os.path.join(dir_to_save_to, filename_to_save + ".png"))
