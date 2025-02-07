from analysis.tree_merger import merge_trees
import numpy as np
from tree.node import TreeNode
from tree.tree import BinaryTree
from set.polyhedron import from_interval
from examples.models.obstacle import example_obstacle

TREE_OUTPUT_FILE = "resources/verified_obstacle/tree.pkl"
TREE_FILES = [
    "resources/verified_obstacle/tree_S1.pkl",
    "resources/verified_obstacle/tree_S2.pkl",
    "resources/verified_obstacle/tree_S3.pkl",
    "resources/verified_obstacle/tree_S4.pkl",
]

def build_obstacle_tree(tree_S1, tree_S2, tree_S3, tree_S4):
    """
    Builds the merged tree using predefined decision nodes for the nonlinear case study.

    Args:
        tree_S1 (BinaryTree): S1 binary tree
        tree_S2 (BinaryTree): S2 binary tree
        tree_S3 (BinaryTree): S3 binary tree
        tree_S4 (BinaryTree): S4 binary tree
    """
    obstacle = example_obstacle()
    min_coords, max_coords = obstacle.get_parameter_domain_interval()

    complete_tree = BinaryTree(from_interval(min_coords, max_coords), tree_S1.output_size)

    decision_node_2 = TreeNode(from_interval(np.array([0.1, 0.0]), np.array([0.7, 1.0])), 2)
    decision_node_2.split(np.array([0., 1.]), -0.4, tree_S2.root, tree_S3.root)

    decision_node_1 = TreeNode(from_interval(np.array([0.1, 0.0]), np.array([1.0, 1.0])), 1)
    decision_node_1.split(np.array([1., 0.]), -0.7, decision_node_2, tree_S4.root)

    complete_tree.root.split(np.array([1., 0.]), -0.1, tree_S1.root, decision_node_1)

    return complete_tree


if __name__ == "__main__":
    merge_trees(TREE_FILES, TREE_OUTPUT_FILE, build_obstacle_tree)
