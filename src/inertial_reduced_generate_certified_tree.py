from argparse import Namespace

import numpy as np

from examples.models.obstacle_inertial import example_obstacle_inertial_reduced
from system.solvable import Solvable
from tree.generator.tree_generator import parse_arguments, run_experiment, save_tree
from set.polyhedron import from_interval

Y_UPPER = 0.39
X_UPPER = 0.7
Y_LOWER = 0.01
X_LOWER = 0.1

SPEED_UPPER = 0.01

bounds = {
    "S1": ([X_LOWER, Y_LOWER, 0, 0], [X_UPPER, Y_UPPER, SPEED_UPPER, SPEED_UPPER]),
    "S2": ([X_LOWER, Y_LOWER, -SPEED_UPPER, 0], [X_UPPER, Y_UPPER, 0, SPEED_UPPER]),
    "S3": ([X_LOWER, Y_LOWER, -SPEED_UPPER, -SPEED_UPPER], [X_UPPER, Y_UPPER, 0, 0]),
    "S4": ([X_LOWER, Y_LOWER, 0, -SPEED_UPPER], [X_UPPER, Y_UPPER, SPEED_UPPER, 0])
}

ZONE_DEFINITIONS = {key: from_interval(np.array(lower), np.array(upper)) for key, (lower, upper) in bounds.items()}

if __name__ == '__main__':
    obstacle: Solvable = example_obstacle_inertial_reduced()
    args: Namespace = parse_arguments()
    chosen_zone = ZONE_DEFINITIONS.get(args.zone, ZONE_DEFINITIONS["S1"])

    tree = run_experiment(args, obstacle, chosen_zone, max_tasks=3000)
    save_tree(tree, args.output, args.seed)
