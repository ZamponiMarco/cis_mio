import joblib
import sys


def load_trees(file_paths):
    """
    Loads multiple trees from given file paths.
    """
    return [joblib.load(path) for path in file_paths]


def compute_tree_statistics(complete_tree, trees):
    """
    Computes and prints final tree statistics (height and data points).
    """
    complete_tree.recompute_height()
    leaves = complete_tree.depth_first_leaves()

    height = max(leaf.height for leaf in leaves)
    points = sum(tree.experiment_results.data_points for tree in trees)

    print(f'Final tree height: {height}')
    print(f'Final tree total points: {points}')


def merge_trees(tree_paths, output_file, build_tree_func):
    """
    Merges individual certified trees using the provided build_tree_func and saves the final tree.
    """
    try:
        trees = load_trees(tree_paths)
    except FileNotFoundError:
        print("Could not find certified tree files")
        sys.exit(1)

    complete_tree = build_tree_func(*trees)

    compute_tree_statistics(complete_tree, trees)

    joblib.dump(complete_tree, output_file)
    print(f'Final tree successfully saved to {output_file}')
