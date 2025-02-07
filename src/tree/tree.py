from typing import Optional, List, Dict

import numpy as np
from matplotlib import pyplot as plt
from set.polyhedron import Polyhedron
from tree.experiment_results import ExperimentResult
from tree.node import TreeNode, VerificationType


class BinaryTree:
    """
    Simple recursive representation of a binary decision tree.

    Attributes:
        initial_polyhedron (Polyhedron): the polyhedron representing the domain of the tree.
        root (Optional[TreeNode]): the root of the tree.
        num_features (int): the number of features in the tree.
        output_size (int): the size of the list of booleans predicted by the tree.
        experiment_results (Optional[ExperimentResult]): the container of metrics concerning the tree construction.
    """
    def __init__(self, initial_polyhedron: Polyhedron, output_size: int) -> None:
        """
        Initialize the binary tree representation.

        Args:
            initial_polyhedron (Polyhedron): the polyhedron representing the domain of the tree.
            output_size (int): the size of the list of booleans predicted by the tree.
        """
        self.initial_polyhedron = initial_polyhedron
        self.root: Optional[TreeNode] = TreeNode(initial_polyhedron, 0)
        self.num_features = initial_polyhedron.A.shape[1]
        self.output_size = output_size
        self.experiment_results: Optional[ExperimentResult] = None

    def fit(self, max_depth: int, splitter: None) -> None:
        """
        Fits the binary tree to the data up to a given depth.

        Args:
            max_depth (int): the maximum depth of the tree.
            splitter: the strategy to determine the best splitting hyperplane.
        """
        self._split_node(self.root, depth=0, max_depth=max_depth, splitter=splitter)

    def predict(self, point: np.ndarray) -> int:
        """
        Predicts the most frequent label at the leaf node for a given point.

        Args:
            point (np.ndarray): the point to be predicted.
        """
        if not self.root:
            raise ValueError("Tree is not initialized.")
        return self._predict_node(self.root, point)

    def pretty_print(self) -> None:
        """Prints the tree structure in a human-readable format."""
        self._print_node(self.root, depth=0)

    def depth_first_leaves(self) -> List[TreeNode]:
        """Returns a list of nodes in depth-first (pre-order) traversal order."""
        return self._collect_leaf(self.root)

    def depth_first_nodes(self) -> List[TreeNode]:
        """Returns a list of nodes in depth-first (pre-order) traversal order."""
        return self._collect_node(self.root)

    def plot_tree(self, ax: plt.Axes, fixed_vars: Optional[Dict[int, float]] = None) -> None:
        """
        Plot the polyhedra for all leaf nodes.

        Args:
            ax (matplotlib.axes.Axes): the axes on which to plot.
            fixed_vars (Optional[Dict[int, float]]): the fixed variables to use for plotting (needed when self.num_features > 2).
        """
        poly = self.initial_polyhedron.fix_variables(fixed_vars=fixed_vars)
        vertices = poly.get_vertices()

        # Adjust the axes limits to include the polyhedron
        x_min, y_min = vertices.min(axis=0)
        x_max, y_max = vertices.max(axis=0)

        # Optionally add some margin around the polyhedron
        margin_x = (x_max - x_min) * 0.1  # 10% margin
        margin_y = (y_max - y_min) * 0.1  # 10% margin
        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(y_min - margin_y, y_max + margin_y)

        self._plot_node(self.root, ax, fixed_vars=fixed_vars)

    def recompute_height(self) -> None:
        """Recursively recomputes the height of each node."""
        if self.root:
            self._recompute_height(self.root, 0)

    def remove_data(self):
        """Removes data from the tree leaves."""
        return self._remove_data(self.root)

    def _collect_leaf(self, node: TreeNode) -> List[TreeNode]:
        """Recursively collects nodes in depth-first order."""
        if not node:
            return []

        nodes = []
        if node.is_leaf:
            nodes = [node]

        if node.left:
            nodes.extend(self._collect_leaf(node.left))
        if node.right:
            nodes.extend(self._collect_leaf(node.right))
        return nodes

    def _collect_node(self, node: TreeNode) -> List[TreeNode]:
        """Recursively collects nodes in depth-first order."""
        if not node:
            return []

        nodes = []

        if node.left:
            nodes.extend(self._collect_node(node.left))
        nodes.extend([self])
        if node.right:
            nodes.extend(self._collect_node(node.right))
        return nodes

    def _split_node(self, node: TreeNode, depth: int, max_depth: int, splitter=None) -> None:
        """Recursively splits nodes until the maximum depth or stopping condition is reached."""
        if depth >= max_depth:
            return

        node.split_once(splitter=splitter)
        if node.left:
            self._split_node(node.left, depth + 1, max_depth, splitter)
        if node.right:
            self._split_node(node.right, depth + 1, max_depth, splitter)

    def _predict_node(self, node: TreeNode, point: np.ndarray) -> list:
        """Recursively traverses the tree to predict the label for a given point."""
        if node.is_leaf:
            return node.value

        if self._should_traverse_left(node, point):
            return self._predict_node(node.left, point)
        else:
            return self._predict_node(node.right, point)

    def _should_traverse_left(self, node: TreeNode, point: np.ndarray) -> bool:
        """Determines whether to traverse left based on the node's decision boundary."""
        return node.split_weights @ point + node.split_bias <= 0

    def _plot_node(self, node: TreeNode, ax: plt.Axes, fixed_vars: Optional[Dict[int, float]] = None) -> None:
        """Recursively plot polyhedra for each node."""
        if node.is_leaf and node.polyhedron:
            node.polyhedron.plot(ax, color='#a1d99b' if node.verified == VerificationType.VERIFIED else '#9ecae1', fixed_vars=fixed_vars)
        if node.left:
            self._plot_node(node.left, ax, fixed_vars=fixed_vars)
        if node.right:
            self._plot_node(node.right, ax, fixed_vars=fixed_vars)

    def _print_node(self, node: TreeNode, depth: int) -> None:
        """Recursively prints the tree structure."""
        indent = "  " * depth
        if node.is_leaf:
            print(f"{indent}Leaf Node [{node.verified}] (depth {depth}), Domain Volume: {node.polyhedron.bounding_box_lengths()}")
        else:
            print(f"{indent}Decision Node (depth {depth}): "
                  f"Split with weights={node.split_weights}, bias={node.split_bias}")
            if node.left:
                print(f"{indent} Left:")
                self._print_node(node.left, depth + 1)
            if node.right:
                print(f"{indent} Right:")
                self._print_node(node.right, depth + 1)

    def _remove_data(self, node: TreeNode):
        """Recursively removes data from the tree leaves."""
        if node.is_leaf:
            node.data = None

        if node.left:
            self._remove_data(node.left)
        if node.right:
            self._remove_data(node.right)

    def _recompute_height(self, node: TreeNode, height: int) -> None:
        """Recursively recomputes node heights"""
        node.height = height

        if node.left:
            self._recompute_height(node.left, height + 1)
        if node.right:
            self._recompute_height(node.right, height + 1)