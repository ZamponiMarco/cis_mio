from math import floor

import numpy as np
import pyDOE3
from set.polyhedron import Polyhedron
from sklearn.gaussian_process import GaussianProcessClassifier
from system.solvable import Solvable
from util.data_util import labels_to_classes
from util.optimization_util import sequential_solve


def sample_points_rejection(seed: int = 0):
    def do(model: Solvable, points: int, domain: Polyhedron, logger=None):
        samples = np.array([
            domain.random_point() for _ in range(points)
        ])
        labels = sequential_solve(model, samples)

        return samples, labels
    return do

def sample_points_lhs(seed: int = 0, model: Solvable = None):
    def do(model: Solvable, points: int, domain: Polyhedron):
        # Find the bounding box limits
        min_coords, max_coords = domain.bounding_box()

        lhs_samples = pyDOE3.lhs(domain.dim, points, random_state=seed)

        # Scale the LHS samples from the unit cube [0, 1] to the bounding box
        scaled_samples = lhs_samples * (max_coords - min_coords) + min_coords

        valid_samples = []

        for sample in scaled_samples:
            if domain.contains(sample):
                valid_samples.append(sample)

        valid_samples = np.array(valid_samples)

        labels = sequential_solve(model, valid_samples)
        return valid_samples, labels
    return do

def sample_points_hit_and_run(seed: int):
    def do(model: Solvable, points: int, domain: Polyhedron, logger=None):
        # Obtain the vertices of the polyhedron
        vertices = domain.get_vertices()  # Assume this returns a numpy array of shape (num_vertices, dimension)

        # Compute the initial point as an equal-weighted linear combination of the vertices
        initial_point = np.mean(vertices, axis=0)
        current_point = initial_point.copy()

        samples = []
        steps_per_sample = 10  # Number of Hit-and-Run steps between recorded samples
        dimension = len(initial_point)

        # Total number of steps to generate the desired number of samples
        total_steps = points * steps_per_sample

        # Ensure the polyhedron is defined by Ax <= b
        A = domain.A  # Constraint matrix of shape (num_constraints, dimension)
        b = domain.b  # Constraint vector of shape (num_constraints,)

        for step in range(total_steps):
            # Generate a random direction uniformly over the unit sphere
            direction = np.random.normal(size=dimension)
            direction /= np.linalg.norm(direction)  # Normalize to unit vector

            # Compute the intersection of the line with the polyhedron
            Ad = A.dot(direction)  # Dot product A * direction
            Ac = A.dot(current_point)  # Dot product A * current_point

            t_min = -np.inf
            t_max = np.inf

            for i in range(len(Ad)):
                if Ad[i] == 0:
                    if Ac[i] > b[i]:
                        # The line is outside the feasible region; no valid t exists
                        t_min = t_max = 0
                        break  # Exit early since no feasible segment exists
                    else:
                        # Constraint does not affect t
                        continue
                else:
                    t_i = (b[i] - Ac[i]) / Ad[i]
                    if Ad[i] > 0:
                        # For Ad[i] > 0, t <= t_i
                        t_max = min(t_max, t_i)
                    else:
                        # For Ad[i] < 0, t >= t_i
                        t_min = max(t_min, t_i)

            if t_min > t_max:
                # No feasible t exists; skip to the next iteration
                continue

            # Sample t uniformly from the feasible interval [t_min, t_max]
            t = np.random.uniform(t_min, t_max)
            next_point = current_point + t * direction
            current_point = next_point

            # Record the sample every 'steps_per_sample' iterations
            if (step + 1) % steps_per_sample == 0:
                samples.append(current_point.copy())

        samples = np.array(samples[:points])  # Ensure we have exactly 'points' samples
        labels = sequential_solve(model, samples)

        return samples, labels
    return do

def sample_active_learning(iterations: int, pool_size: int, seed: int = 0):
    def adaptive_sampling_uncertainty(model: Solvable, points: int, domain: Polyhedron, logger=None):
        pool = np.array([
            domain.random_point() for _ in range(pool_size)
        ])

        to_do = points
        starting_points = floor(points/iterations)
        samples = np.array([
            domain.random_point() for _ in range(starting_points)
        ])
        labels = sequential_solve(model, samples)
        classes = labels_to_classes(labels)
        to_do -= starting_points

        clf = GaussianProcessClassifier()
        if len(np.unique(labels_to_classes(labels))) > 1:
            clf.fit(samples, classes)

        while to_do > 0:
            to_take = floor(points/iterations)
            if to_take >= to_do:
                to_take = to_do
            to_do -= to_take

            if hasattr(clf, "classes_"):
                probs = clf.predict_proba(pool)

                # Compute entropy for each point
                entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

                sampling_distribution = entropy / np.sum(entropy)
            else:
                sampling_distribution = np.ones(len(pool)) / len(pool)

            selected_indices = np.random.choice(
                len(pool),
                size=to_take,
                replace=False,
                p=sampling_distribution
            )
            new_X = pool[selected_indices]
            new_y = sequential_solve(model, new_X)

            if len(np.unique(labels_to_classes(new_y))) > 1:
                clf.fit(new_X, labels_to_classes(new_y))

            samples = np.vstack([
                samples, new_X
            ])
            labels = np.vstack([
                labels, new_y
            ])
            pool = np.delete(pool, selected_indices, axis=0)

        return samples, labels
    return adaptive_sampling_uncertainty