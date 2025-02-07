import numpy as np

from examples.models.obstacle import example_obstacle
from tree.generator.tree_generator import parse_arguments, run_experiment, save_tree
from set.polyhedron import from_interval

bounds = {
    "S1": ([0.0, 0.0], [0.1, 1.0]),
    "S2": ([0.1, 0.0], [0.7, 0.4]),
    "S3": ([0.1, 0.5], [0.7, 1.0]),
    "S4": ([0.7, 0.0], [1.0, 1.0])
}

ZONE_DEFINITIONS = {key: from_interval(np.array(lower), np.array(upper)) for key, (lower, upper) in bounds.items()}

if __name__ == '__main__':
    obstacle = example_obstacle()
    args = parse_arguments()
    chosen_zone = ZONE_DEFINITIONS.get(args.zone, ZONE_DEFINITIONS["S1"])

    tree = run_experiment(args, obstacle, chosen_zone, max_tasks=1000)
    save_tree(tree, args.output, args.seed)
