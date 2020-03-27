# euler-meets-cuda
## Bridges
### Testing
To run automatic tests
```shell
    ./bridges_test.sh -o bridgesResult.txt
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
To run automatic tests
```shell
    ./lca_test.sh -t
```
Time results will be saved to ```tests/lca/resultTimes``` (WIP)

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