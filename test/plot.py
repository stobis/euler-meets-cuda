import matplotlib as mpl
mpl.use('Agg')

import re
import sys
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


# Here we declare what tests consist an experiment.
# An experiment is described by a regex that should be contained in input's filename
experiments = ['E1', 'E2', 'E3', 'road|osm', 'kron',
               'citationCiteseer|com-LiveJournal|hollywood|Stanford']
experiments_use_filename = [False, False, False, True, True, True]

csv_filename = "file"
csv_N = 'N'
csv_algo = 'algo'
csv_overall = 'Overall'
csv_parameters = [csv_N, 'grasp', 'batch']

csv_ints = []
csv_ints.extend(csv_parameters)
csv_floats = []
csv_floats.append(csv_overall)

with open(sys.argv[1], 'r') as file:
    csv_reader = csv.DictReader(file)

    csv_file = []
    for line in csv_reader:
        csv_file.append(line)

    for i_exp, experiment in enumerate(experiments):
        plots = []
        name_sufix = []

        current_rows = []
        for row in csv_file:
            if bool(re.search(experiment, row[csv_filename])):
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

            for i, row in enumerate(algo_rows):
                for param in csv_ints:
                    if param in row:
                        row[param] = int(float(row[param]))
                for param in csv_floats:
                    if param in row:
                        row[param] = float(row[param])
                algo_rows[i] = row

            parameter_values = {}
            for param in csv_parameters:
                parameter_values[param] = set()

            for row in algo_rows:
                for param in csv_parameters:
                    if param in row:
                        val = row[param]
                        if val not in parameter_values[param]:
                            parameter_values[param].add(val)

            print(parameter_values)

            parameter_sizes = {}
            for param in csv_parameters:
                parameter_sizes[param] = len(parameter_values[param])

            # For each test there are different params (N, batch, grasp). On X axis will be param with most values
            param_to_plot_by = max(parameter_sizes, key=parameter_sizes.get)
            print(param_to_plot_by)

            # For now we allow one additional parameter to plot (we produce a plot for each of its values)
            parameter_sizes[param_to_plot_by] = -1
            additional_param = max(parameter_sizes, key=parameter_sizes.get)
            additional_param_values = parameter_values[additional_param]
            if parameter_sizes[additional_param] <= 1:
                additional_param = ""
                additional_param_values = [""]

            print(additional_param)
            print(additional_param_values)

            x = sorted(parameter_values[param_to_plot_by])
            print(x)

            for i, a_param in enumerate(additional_param_values):
                y = [''] * len(x)

                for row in algo_rows:
                    for i_xval, xval in enumerate(x):
                        if row[param_to_plot_by] == xval and (additional_param == '' or row[additional_param] == a_param):
                            y[i_xval] = row[csv_overall] / 1000
                            if experiments_use_filename[i_exp]:
                                x[i_xval] = row[csv_filename]

                if len(plots) <= i:
                    plots.append(plt.subplots())
                    # plots.append(plt.subplots(constrained_layout=True))
                    name_sufix.append(
                        "" if additional_param == '' else additional_param + "=" + str(a_param))
                fig, ax = plots[i]
                ax.set_xlabel(param_to_plot_by)
                ax.set_ylabel("Time overall (s)")

                if experiments_use_filename[i_exp]:
                    width = 0.3
                    real_x = np.arange(len(x))
                    bar_shift = (i_algo + 0.5 - (len(algos) / 2.0))*width
                    ax.bar(real_x + bar_shift, y, width, label = algo)
                    ax.set_xticklabels(x)
                    ax.set_xticks(real_x)
                else:
                    ax.plot(x, y, label=algo)

        for i, (fig, ax) in enumerate(plots):
            ax.legend()
            if experiments_use_filename[i_exp]:
                plt.xticks(rotation=30, ha='right')
                # fig.subplots_adjust(bottom=0.2)
                fig.tight_layout()
            else:
                ax.set_xscale("log")
                # fig.constrained_layout()
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ScalarFormatter())
            fig.savefig(experiment + name_sufix[i] + ".png")
