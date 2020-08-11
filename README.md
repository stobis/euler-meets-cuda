# euler-meets-cuda

## Contents
Headers are located in ```include/``` with sources in ```src/```
- ```include/euler.h``` and ```src/euler.cu``` contain implementation of the Euler Tour and list rank algorithm.
- ```include/bridges.h``` and ```src/bridges.cu``` contain implementation of bridges.
- ```include/lca.h``` and ```src/lca.cu``` contain implementation of lca.

Header files of all the above contain an explanation of input parameters along with a simple input and output.

```./bridges_runner.cu``` and ```./lca_runner.cu``` are working examples use of how to use the methods above in a project.


## Cloning and building instructions
If you are cloning this from the anonymized github link, you may use [clone-anonymous4open](https://github.com/ShoufaChen/clone-anonymous4open):
- ```git clone https://github.com/ShoufaChen/clone-anonymous4open```
- ```cd clone-anonymous4open```
- ```python3 clone.py --clone-dir ../euler-meets-GPU  --target [https://anonymous.4open.science/repository/...]```
- ```cd ../euler-meets-GPU```
- ```git clone https://github.com/moderngpu/moderngpu.git 3rdparty/moderngpu```
- Fix Makefile before building: ```sed -i 's/\[.*\]/$@/g' Makefile```

You may wish to update Makefile variables: CUDA, NVCC and you GPU's computing capability (NVCCSM) to match your system before building.

In case of stack overflow problems (e.g. segfaults when generating tests)
```shell
    ulimit -s unlimited
```

### Bridges
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

### Lca
To build and run automatic tests (be advised that default tests take a couple of hours to complete, modify ```test/lca/testVariables.sh``` to run a subset)
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


### Generating plots
Script generating plots requires matplotlib, run ```pip2 install matplotlib``` to install it.

```shell
    python2 test/plot.py {lca,bridges}Result.csv
```
By default the plots will be saved to ```testResults/plots```
