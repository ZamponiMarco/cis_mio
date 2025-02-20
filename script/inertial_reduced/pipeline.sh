#!/bin/bash

set -e

cd ../..

python3 src/inertial_reduced_analysis_check_satisfying_trees_best.py
python3 src/inertial_reduced_construct_tree.py
python3 src/inertial_reduced_generate_baseline_tree.py
python3 src/inertial_reduced_generate_timings.py
python3 src/inertial_reduced_graph_comparison_times.py
python3 src/inertial_reduced_analysis_comparison.py