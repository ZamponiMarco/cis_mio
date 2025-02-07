#!/bin/bash

set -e

cd ../..

python3 src/obstacle_analysis_check_satisfying_trees_best.py
python3 src/obstacle_construct_tree.py
python3 src/obstacle_generate_baseline_tree.py
python3 src/obstacle_generate_timings.py
python3 src/obstacle_graph_comparison_times.py
python3 src/obstacle_analysis_comparison.py