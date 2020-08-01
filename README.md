# euler-meets-cuda

In case of stack overflow problems (e.g. segfaults when generating tests)
```shell
    ulimit -s unlimited
```

## Bridges
To build, run automatic tests and generate plots
```shell
    ./bridges_test.sh -t bridgesResult.csv -p
```

To generate plots from csv with results
```shell
    python2 test/plot.py testResults/bridgesResult.csv
```

To run single test
```shell
    make bridges_runner
    ./bridges_runner.e -i [input file] -o [output file] -a [algorithm to use]
```
For help run ```./bridges_runner.e -h```




## Lca
To build, run automatic tests and generate plots (be advised that default tests take about 10h to complete, modify ```test/lca/testVariables.sh``` to run a subset)
```shell
    ./lca_test.sh -t lcaResult.csv -p
```

To generate plots from csv with results
```shell
    python2 test/plot.py testResults/lcaResult.csv
```

To run automatic tests checking correctness
```shell
    ./lca_test.sh -c
```

To run single test
```shell
    make lca_runner.e
    ./lca_runner.e -i [input file] -o [output file] -a [algorithm to use]
```
For help run ```./lca_runner.e -h```
