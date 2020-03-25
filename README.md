# euler-meets-cuda
## Bridges
### Testing
The simpliest example
```shell
    ulimit -s unlimited
```
```shell
    make
    cd test/bridges
    make
    ./prepare.py # urls are stored in config.yml
    ./run.sh
```

With export to csv:
```shell
    ...
    ./run_and_export.sh [naive|tarjan|hybrid] [output] # algorithm must be specified for now
```

## Lca
To run tests checking correctness of algorithms
```shell
    ./lca_test.sh -c
```

To run timed tests and export times (WIP)
```shell
    ./lca_test.sh -t
```