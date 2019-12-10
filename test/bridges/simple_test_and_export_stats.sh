#!/bin/bash

./simple_test_loop.sh > stats.out
python3 export_results.py stats.out stats.csv
