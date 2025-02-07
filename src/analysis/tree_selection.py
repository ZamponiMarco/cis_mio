import os
import joblib
import sys
from typing import Optional
from tree.tree import BinaryTree


def get_best_tree(directory: str) -> Optional[BinaryTree]:
    """
    Finds the best certified tree in the given directory containing a tree collection for a subdomain.

    Args:
        directory (str): path to the directory containing the tree collection.
    """
    best_tree: Optional[BinaryTree] = None

    for file in os.listdir(directory):
        tree: BinaryTree = joblib.load(os.path.join(directory, file))
        tree_res = tree.experiment_results

        if tree_res.verified_leaves != tree_res.total_leaves:
            continue

        if not best_tree:
            best_tree = tree
            continue

        best_tree_res = best_tree.experiment_results

        if tree_res.max_height < best_tree_res.max_height or \
                (tree_res.max_height == best_tree_res.max_height and tree_res.total_nodes < best_tree_res.total_nodes):
            best_tree = tree

    return best_tree


def process_directories(base_source: str, base_target: str, directories: list[str]):
    """
    Processes directories containing tree collections, finding and saving the best tree in each in the target directory.

    Args:
        base_source (str): base directory containing the tree collections.
        base_target (str): directory where the best trees are saved.
        directories (str): list of subdirectories of source directory to process.
    """
    os.makedirs(base_target, exist_ok=True)

    print("Searching for best certified trees.")

    try:
        for directory in directories:
            source_path = os.path.join(base_source, directory)
            print(f"Processing directory: {directory} - ", end='')
            best_tree = get_best_tree(source_path)
            print(f"Finished processing {directory} - ", end='')

            if best_tree:
                joblib.dump(best_tree, os.path.join(base_target, f'tree_{directory}.pkl'))
                print(f"Best tree for {directory} saved successfully.")
            else:
                print(f"No valid tree found for {directory}.")
    except FileNotFoundError:
        print("Tree files not found, was the tree construction script run?")
        sys.exit(1)

