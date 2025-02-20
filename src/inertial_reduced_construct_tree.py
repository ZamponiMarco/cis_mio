import numpy as np

from analysis.tree_merger import merge_trees
from set.polyhedron import from_interval
from tree.node import TreeNode
from tree.tree import BinaryTree

TREE_OUTPUT_FILE = "resources/verified_inertial_reduced/tree.pkl"
TREE_FILES = [
    "resources/verified_inertial_reduced/tree_S1.pkl",
    "resources/verified_inertial_reduced/tree_S2.pkl",
    "resources/verified_inertial_reduced/tree_S3.pkl",
    "resources/verified_inertial_reduced/tree_S4.pkl",
]

def build_inertial_tree(tree_S1, tree_S2, tree_S3, tree_S4):
    """
    Builds the merged tree using predefined decision nodes for the inertial case study.

    Args:
        tree_S1 (BinaryTree): S1 binary tree.
        tree_S2 (BinaryTree): S2 binary tree.
        tree_S3 (BinaryTree): S3 binary tree.
        tree_S4 (BinaryTree): S4 binary tree.
    """
    min_coords = np.array([0.1, 0.01, -0.01, -0.01])
    max_coords = np.array([0.7, 0.39, 0.01, 0.01])

    complete_tree = BinaryTree(from_interval(min_coords, max_coords), tree_S1.output_size)

    decision_node_1 = TreeNode(from_interval(np.array([0.1, 0.01, 0, -0.01]), max_coords), 1)
    decision_node_1.split(np.array([0, 0, 0, 1]), 0, tree_S4.root, tree_S1.root)

    decision_node_2 = TreeNode(from_interval(min_coords, np.array([0.7, 0.39, 0, 0.01])), 1)
    decision_node_2.split(np.array([0, 0, 0, 1]), 0, tree_S3.root, tree_S2.root)

    complete_tree.root.split(np.array([0, 0, 1, 0]), 0, decision_node_2, decision_node_1)

    return complete_tree


if __name__ == "__main__":
    merge_trees(TREE_FILES, TREE_OUTPUT_FILE, build_inertial_tree)

