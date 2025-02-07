from typing import Tuple, Optional

import numpy as np
from set.polyhedron import Polyhedron
from system.solvable import Solvable


def half_split():
    def do(X: np.ndarray, y: np.ndarray, polyhedron: 'Polyhedron') -> Tuple[np.ndarray, float] or None:
        lower_bounds, upper_bounds = polyhedron.bounding_box()
        longest_feature = np.argmax(upper_bounds - lower_bounds)
        best_feature = longest_feature
        threshold = (upper_bounds[longest_feature] + lower_bounds[longest_feature]) / 2

        n_features = X.shape[1]

        # Construct weight vector and bias
        w = np.zeros(n_features)
        w[best_feature] = 1
        b = -threshold

        return w, b

    return do


def find_best_split_decision_tree(seed: int = 0, model: Solvable = None):
    def gini_index(output_points: np.ndarray) -> float:
        _, counts = np.unique(output_points, return_counts=True, axis=0)
        total = sum(counts)
        return 1 - sum([(count / total) ** 2 for count in counts])

    def get_best_split(X: np.ndarray, y: np.ndarray, polyhedron: 'Polyhedron') -> Tuple[Optional[int], Optional[float]]:
        best_feature, best_value, best_score = None, None, float('inf')
        lower_bounds, upper_bounds = polyhedron.bounding_box()
        for feature in range(X.shape[1]):
            # if model is not None \
            #         and model.discretize(lower_bounds)[feature] == model.discretize(upper_bounds)[feature]:
            #     continue
            unique_values = np.unique(X[:, feature])
            for i in range(len(unique_values) - 1):
                value = (unique_values[i] + unique_values[i + 1]) / 2
                left_indices = X[:, feature] < value
                right_indices = X[:, feature] >= value
                left_gini = gini_index(y[left_indices])
                right_gini = gini_index(y[right_indices])
                left_weight = sum(left_indices) / len(y)
                right_weight = sum(right_indices) / len(y)
                gini = left_weight * left_gini + right_weight * right_gini

                if gini < best_score:
                    best_feature, best_value, best_score = feature, value, gini

        if best_feature is None:
            longest_feature = np.argmax(upper_bounds - lower_bounds)
            best_feature = longest_feature
            best_value = (upper_bounds[longest_feature] + lower_bounds[longest_feature]) / 2

        return best_feature, best_value

    def do(X: np.ndarray, y: np.ndarray, polyhedron: 'Polyhedron') -> Tuple[np.ndarray, float] or None:
        # Get the best split using CART algorithm
        best_feature, threshold = get_best_split(X, y, polyhedron)
        if best_feature is None:
            return None
        n_features = X.shape[1]

        # Construct weight vector and bias
        w = np.zeros(n_features)
        w[best_feature] = 1
        b = -threshold

        return w, b

    return do


def get_majority_class(y: np.ndarray) -> int:
    """
    Finds the class label with the most occurrences.
    """
    values, counts = np.unique(y, return_counts=True)
    majority_class = values[np.argmax(counts)]
    return majority_class


def split_polyhedron(polyhedron: Polyhedron, weights: np.ndarray, bias: float) -> Tuple[
    Polyhedron, Polyhedron]:
    """
    Split the polyhedron into left and right using the linear function weights*x + bias <= 0.
    """
    left_polyhedron = Polyhedron(polyhedron.A, polyhedron.b).add_halfspace(weights, -bias)
    right_polyhedron = Polyhedron(polyhedron.A, polyhedron.b).add_halfspace(-weights, bias)

    return left_polyhedron, right_polyhedron
