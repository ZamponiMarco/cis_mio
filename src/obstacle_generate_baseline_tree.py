import joblib
import sys

from analysis.train_model import train_and_save_model
from generate_problem_data import generate_problem_data
from examples.models.obstacle import example_obstacle

POINTS = 500

PROBLEM_DATA_OUTPUT_FILE = 'resources/verified_obstacle/problem_data.csv'
VERIFIED_TREE_FILE = 'resources/verified_obstacle/tree.pkl'

SIMPLE_TREE_OUTPUT_FILE = 'resources/verified_obstacle/simple_tree.pkl'
FOREST_TREE_OUTPUT_FILE = 'resources/verified_obstacle/forest_tree.pkl'

FEATURES = ['theta_0', 'theta_1']

if __name__ == '__main__':
    obstacle = example_obstacle()

    try:
        tree = joblib.load(VERIFIED_TREE_FILE)
    except FileNotFoundError:
        print("Could not find final merged tree.")
        sys.exit(1)

    root_polyhedron = tree.initial_polyhedron
    output_size = tree.output_size

    generate_problem_data(obstacle, root_polyhedron, output_size, PROBLEM_DATA_OUTPUT_FILE, POINTS)

    train_and_save_model(PROBLEM_DATA_OUTPUT_FILE, 'decision_tree', SIMPLE_TREE_OUTPUT_FILE, FEATURES, max_depth=8)

    generate_problem_data(obstacle, root_polyhedron, output_size, PROBLEM_DATA_OUTPUT_FILE, POINTS)

    train_and_save_model(PROBLEM_DATA_OUTPUT_FILE, 'random_forest', FOREST_TREE_OUTPUT_FILE, FEATURES, max_depth=8, n_estimators=10)
