import copy
from typing import List, Callable

from util.tree_util import *

from enum import Enum

class VerificationType(Enum):
    """
    Possible verification statuses for each leaf node.

    Attributes:
        UNVERIFIED (int): the leaf in not yet verified.
        VERIFIED (int): the lead has been verified.
        TOLERANCE_VERIFIED (int): the leaf domain is smaller than a tolerance parameter and is considered verified.
    """
    UNVERIFIED = 0
    VERIFIED = 1
    TOLERANCE_VERIFIED = 2

class TreeNode:
    """
    A class representing a node in a decision tree structure. Each node can be either a leaf or a decision node.

    In the case of a leaf node it contains details about its corresponding polyhedral domain and the data points
    used to infer the predicted value.

    In the case of a decision node it contains information about the splitting hyperplane and the left and right
    children.

    Attributes:
        is_leaf (bool): whether the node is a leaf node or not.
        height (int): the node depth in the tree.

        split_weights (Optional[np.ndarray]): the linear weights representing the splitting hyperplane.
        split_bias (Optional[float]): the constant coefficient representing the splitting hyperplane.
        left (Optional[TreeNode]): left child of the node.
        right (Optional[TreeNode]): right child of the node.

        polyhedron (Optional[Polyhedron]): polyhedron represented by the leaf node.
        data (Optional[Tuple[np.ndarray, np.ndarray]]): data points couples used to define predicted value.
        value (Optional[List[bool]]): the predicted value.
        verified (VerificationType): verification status of the leaf node.
    """
    def __init__(self, polyhedron: Polyhedron, height: int) -> None:
        """
        Initialize a TreeNode with a polyhedron and a specific height. Nodes are initialized as leaf nodes.

        Args:
            polyhedron (Polyhedron): polyhedron represented by the leaf node.
            height (int): the node depth in the tree.
        """
        self.is_leaf: bool = True  # Initially, each node is a leaf
        self.height: int = height

        self.split_weights: Optional[np.ndarray] = None
        self.split_bias: Optional[float] = None
        self.left: Optional[TreeNode] = None
        self.right: Optional[TreeNode] = None

        self.polyhedron: Optional[Polyhedron] = polyhedron
        self.data: Optional[Tuple[np.ndarray, np.ndarray]] = (np.array([]).reshape(0, 0), np.array([]))  # Empty dataset
        self.value: Optional[List[bool]] = None  # Predicted value
        self.verified: VerificationType = VerificationType.UNVERIFIED  # Initial verification status

    def update_from_node(self, other: 'TreeNode') -> None:
        """
        Updates the current TreeNode's attributes with those from another TreeNode instance.

        Args:
            other (TreeNode): instance containing the data bein copied.
        """
        for attr, value in other.__dict__.items():
            setattr(self, attr, copy.deepcopy(value))

    def split_once(
            self,
            splitter: Callable[[np.ndarray, np.ndarray, Polyhedron], Tuple[np.ndarray, float]]=find_best_split_decision_tree,
            fallback_splitter=None
    ) -> bool:
        """
        Splits the current leaf node into two child nodes using the given splitter technique.

        Args:
            splitter (Callable[[np.ndarray, np.ndarray, Polyhedron], Tuple[np.ndarray, float]]): the splitter technique to use.
            fallback_splitter: fallback splitter technique to use.
        """

        if not self.is_leaf:  # If node has child return
            return False

        X, y = self.data

        if len(np.unique(y)) < 2:  # If there are less than two data points
            if fallback_splitter:
                split = fallback_splitter(X, y, self.polyhedron)
            else:
                return False
        else:
            # Find a linear split (weights, bias)
            split = splitter(X, y, self.polyhedron)

            # If no split is found split by half of longest feature
            if split is None:
                if fallback_splitter:
                    split = fallback_splitter(X, y, self.polyhedron)
                else:
                    return False

        weights, bias = split

        children = self._generate_child_nodes(weights, bias)

        if not children:  # If children generation was not successful return
            return False

        left_node, right_node = children
        self.split(weights, bias, left_node, right_node)
        return True

    def split(self, weights: np.ndarray, bias: float, left_node: 'TreeNode', right_node: 'TreeNode') -> None:
        """
        Splits the current node into the two children nodes passed as arguments.

        Args:
            weights (np.ndarray): the linear coefficients representing the splitting hyperplane.
            bias (float): the constant coefficient representing the splitting hyperplane.
            left_node (TreeNode): left child of the node.
            right_node (TreeNode): right child of the node.
        """

        self.is_leaf = False

        # Set Decision Node Values
        self.split_weights = weights
        self.split_bias = bias
        self.left = left_node
        self.right = right_node

        # Unset Leaf Node Values
        self.polyhedron = None
        self.data = None
        self.value = None

    def add_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Adds data to the current node. If it's a leaf, store the data, otherwise pass to child nodes.

        Args:
            X (np.ndarray): input data.
            y (np.ndarray): predicted values.
        """
        if self.is_leaf:
            self._store_leaf_data(X, y)
        else:
            self._distribute_data(X, y)

    def _generate_child_nodes(self, weights: np.ndarray, bias: float) -> Optional[Tuple['TreeNode', 'TreeNode']]:
        X, y = self.data

        # Split data using the linear combination
        left_mask = (X @ weights + bias) <= 0
        right_mask = ~left_mask
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        # If one of the resulting dataset is empty do not split
        if len(X_left) == 0 or len(X_right) == 0:
            return None

        # Split Polyhedron
        left_polyhedron, right_polyhedron = split_polyhedron(self.polyhedron, weights, bias)

        # If volume of resulting polyhedra is 0 do not split
        if left_polyhedron.volume() == 0 or right_polyhedron.volume() == 0:
            return None

        # Create children nodes and add corresponding data points
        left_node = TreeNode(left_polyhedron, self.height + 1)
        right_node = TreeNode(right_polyhedron, self.height + 1)
        left_node.add_data(X_left, y_left)
        right_node.add_data(X_right, y_right)

        return left_node, right_node

    def _store_leaf_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Stores data in a leaf node and computes the most frequent label."""
        if len(self.data[0]) == 0:
            self.data = (X, y)
        else:
            self.data = (np.vstack([self.data[0], X]), np.hstack([self.data[1], y]))

        if len(self.data[1]) > 0:
            self.value = get_majority_class(self.data[1])

    def _distribute_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Distributes data recursively to the left or right child nodes based on the split."""
        left_mask = (X @ self.split_weights + self.split_bias) <= 0
        right_mask = ~left_mask

        if np.any(left_mask):
            self.left.add_data(X[left_mask], y[left_mask])
        if np.any(right_mask):
            self.right.add_data(X[right_mask], y[right_mask])
