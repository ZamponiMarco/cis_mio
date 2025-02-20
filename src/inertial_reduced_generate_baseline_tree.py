import sys

import joblib

from analysis.train_model import train_and_save_model
from generate_problem_data import generate_problem_data
from examples.models.obstacle_inertial import example_obstacle_inertial_reduced

POINTS = 1000

PROBLEM_DATA_OUTPUT_FILE = 'resources/verified_inertial_reduced/problem_data.csv'
VERIFIED_TREE_FILE = 'resources/verified_inertial_reduced/tree.pkl'

SIMPLE_TREE_OUTPUT_FILE = 'resources/verified_inertial_reduced/simple_tree.pkl'
FOREST_TREE_OUTPUT_FILE = 'resources/verified_inertial_reduced/forest_tree.pkl'

FEATURES = ['theta_0', 'theta_1', 'theta_2', 'theta_3']

if __name__ == '__main__':
    obstacle = example_obstacle_inertial_reduced()

    try:
        tree = joblib.load(VERIFIED_TREE_FILE)
    except FileNotFoundError:
        print("Could not find final merged tree.")
        sys.exit(1)

    root_polyhedron = tree.initial_polyhedron
    output_size = tree.output_size

    generate_problem_data(obstacle, root_polyhedron, output_size, PROBLEM_DATA_OUTPUT_FILE, POINTS)

    train_and_save_model(PROBLEM_DATA_OUTPUT_FILE, 'decision_tree', SIMPLE_TREE_OUTPUT_FILE, FEATURES, max_depth=10)

    generate_problem_data(obstacle, root_polyhedron, output_size, PROBLEM_DATA_OUTPUT_FILE, POINTS)

    train_and_save_model(PROBLEM_DATA_OUTPUT_FILE, 'random_forest', FOREST_TREE_OUTPUT_FILE, FEATURES, max_depth=10, n_estimators=10)

