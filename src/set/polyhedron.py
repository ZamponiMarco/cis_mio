from typing import Dict, Optional, Tuple, List

import cdd
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull, QhullError
from z3 import And, ArithRef


def from_interval(min, max):
    # Convert the interval representation to half-space representation A*x <= b
    # For each interval, create two inequalities: x >= lower_bound and x <= upper_bound
    dim = len(min)
    A = np.vstack([np.eye(dim), -np.eye(dim)])  # A for x <= upper_bound and -x <= -lower_bound
    b = np.hstack([max, -min])

    return Polyhedron(A, b)


class Polyhedron:
    def __init__(self, A: np.ndarray, b: np.ndarray) -> None:
        self.dim = A.shape[1]

        h_representation = np.hstack([b.reshape(-1, 1), -A])
        mat = cdd.Matrix(h_representation, number_type='fraction')
        mat.rep_type = cdd.RepType.INEQUALITY
        mat.canonicalize()

        # Create a cdd Polyhedron
        cdd_polyhedron = cdd.Polyhedron(mat)

        generators = cdd_polyhedron.get_generators()
        vertices = []
        for row in generators:
            if row[0] == 1:  # 1 indicates a vertex
                vertices.append(row[1:])
        self.vertices = np.array(vertices)

        inequalities = cdd_polyhedron.get_inequalities()
        self.A = -np.array(inequalities)[:, 1:]
        self.b = np.array(inequalities)[:, 0]

    def add_halfspace(self, a: np.ndarray, b_val: float) -> 'Polyhedron':
        new_A = np.vstack([self.A, a])
        new_b = np.hstack([self.b, b_val])
        return Polyhedron(new_A, new_b)

    def intersect_polyhedra(self, polyhedron: 'Polyhedron') -> 'Polyhedron':
        A_combined = np.vstack((self.A, polyhedron.A))
        b_combined = np.hstack((self.b, polyhedron.b))
        return Polyhedron(A_combined, b_combined)

    def fix_variables(self, fixed_vars: Dict[int, float]) -> 'Polyhedron':
        fixed_indices = list(fixed_vars.keys())
        free_indices = [i for i in range(self.dim) if i not in fixed_indices]

        A_free = self.A[:, free_indices]
        A_fixed = self.A[:, fixed_indices]

        b_adjusted = self.b - A_fixed @ np.array([fixed_vars[i] for i in fixed_indices])

        return Polyhedron(A_free, b_adjusted)

    def contains(self, x: np.ndarray) -> bool:
        return np.all(self.A @ x <= self.b)

    def contains_symbolic(self, point_symbols: List[ArithRef]) -> And:
        if self.A.size == 0 or len(point_symbols) != self.dim:
            raise ValueError("Mismatch between point dimensions and polyhedron constraints.")

        constraints = [
            sum(self.A[i, j] * point_symbols[j] for j in range(self.dim)) <= self.b[i]
            for i in range(self.A.shape[0])
        ]
        return And(*constraints)

    def get_vertices(self) -> np.ndarray:
        return self.vertices

    def volume(self) -> float:
        if self.vertices.size == 0:
            return 0

        try:
            hull = ConvexHull(self.vertices)
            return hull.volume
        except (QhullError, IndexError):
            return 0

    def random_point(self, max_attempts=1000, distributions: Optional[list] = None):
        min_coords, max_coords = self.bounding_box()

        bounds_min = min_coords.astype(float)
        bounds_max = max_coords.astype(float)

        for _ in range(max_attempts):
            if distributions:
                point = [distribution() for distribution in distributions]
            else:
                point = np.random.uniform(low=bounds_min, high=bounds_max, size=self.dim)

            if self.contains(point):
                return point

        raise ValueError("Failed to find a point within the polyhedron after several attempts.")

    def plot(self, ax: plt.Axes, color: str = 'blue', fixed_vars: Optional[Dict[int, float]] = None) -> None:
        if self.dim > 2:
            if fixed_vars is None or len(fixed_vars) != self.dim - 2:
                raise ValueError("For dimensions > 2, please provide fixed_vars to reduce dimension to 2.")
            reduced_polyhedron = self.fix_variables(fixed_vars)
            reduced_polyhedron.plot(ax, color)
        else:
            vertices = self.get_vertices()
            if vertices.shape[0] > 0:
                hull = ConvexHull(vertices)
                polygon = plt.Polygon(vertices[hull.vertices], closed=True, facecolor=color, edgecolor='black',
                                      alpha=0.7)
                ax.add_patch(polygon)

    def bounding_box_lengths(self) -> np.ndarray:
        min_coords, max_coords = self.bounding_box()
        lengths = max_coords - min_coords
        return lengths

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        vertices = self.vertices

        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)

        return min_coords, max_coords

    def __str__(self) -> str:
        num_hyperplanes = self.A.shape[0]

        if num_hyperplanes > 0:
            sample_constraint = " + ".join([f"{coef:.2f}*x{j}" for j, coef in enumerate(self.A[0])])
            sample_constraint += f" <= {self.b[0]:.2f}"
        else:
            sample_constraint = "No constraints"

        return f"Polyhedron with {num_hyperplanes} hyperplanes, sample: ({sample_constraint})"
