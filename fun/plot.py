import matplotlib as mpl
mpl.use('Agg')

import sys
import csv
import matplotlib.pyplot as plt
import numpy as np


experiments = ['E1', 'E2', 'E3']

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

    for experiment in experiments:
        plots = []
        name_sufix = []

        print(experiment + ": ")
        current_rows = []
        for row in filter((lambda x: True if experiment in x[csv_filename] else False), csv_file):
            current_rows.append(row)

        algos = set()
        for row in current_rows:
            if row[csv_algo] not in algos:
                algos.add(row[csv_algo])

        for algo in algos:
            algo_rows = filter(
                lambda row: True if row[csv_algo] == algo else False, current_rows)

            for i, row in enumerate(algo_rows):
                for param in csv_ints:
                    row[param] = int(float(row[param]))
                for param in csv_floats:
                    row[param] = float(row[param])
                algo_rows[i] = row

            parameter_values = {}
            for param in csv_parameters:
                parameter_values[param] = set()

            for row in algo_rows:
                for param in csv_parameters:
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
                            y[i_xval] = row[csv_overall]

                if len(plots) <= i:
                    plots.append(plt.subplots())
                    name_sufix.append(
                        "" if additional_param == '' else additional_param + "=" + str(a_param))
                fig, ax = plots[-1]

            ax.plot(x, y, label=algo)

        for i, (fig, ax) in enumerate(plots):
            ax.legend()
            plt.xscale("log")
            plt.yscale("log")
            fig.savefig(experiment + name_sufix[i] + ".png")

        #     x = []
        #     y = []
        #     for row in algo_rows:
        #         x.append(float(row[csv_N]))
        #         y.append(float(row[csv_overall]))
        #     ax.plot(x, y, label=algo)

        # fig.savefig(experiment + ".png")


# for line in fileinput.input():
#   print("AAA: " + line)

# fig, ax = plt.subplots()  # Create a figure containing a single axes.
# ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

# fig.savefig('foo.png')
