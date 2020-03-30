# euler-meets-cuda
## Bridges
To build and run automatic tests
```shell
    ./bridges_test.sh -t bridgesResult.csv
```

To run single test
```shell
    make bridges_runner
    ./bridges_runner.e -i [input file] -o [output file] -a [algorithm to use]
```
For help run ```./bridges_runner.e -h```

In case of stack overflow problems
```shell
    ulimit -s unlimited
```


## Lca
To build and run automatic tests
```shell
    ./lca_test.sh -t lcaResult.csv
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
