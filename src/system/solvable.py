from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from set.polyhedron import Polyhedron
from z3 import Context, Solver, Real


class Solvable(ABC):
    """
    The Solvable class represents a multi parametric Mixed-Integer Optimization (mp-MIO) problem.
    By extending this class more case studies can be added seamlessly to the package.
    """

    @abstractmethod
    def solve(self, point: np.ndarray):
        """
        Solves the optimization problem for a given point (assignment of parameter values theta).
        This method is expected to return an object containing the computed solution to the optimization problem. The
        returned object is also supposed to be fed to the self.get_output(*) method to retrieve the assignment of
        boolean values.

        Args:
            point (np.ndarray): assignment of parameters for which we are solving the problem.
        """
        raise NotImplementedError()

    @abstractmethod
    def solve_fixed(self, point: np.ndarray, fixed_delta: List[bool]):
        """
        Solves the optimization problem for a given point (assignment of parameters theta) and a given assignment of
        boolean values. This method is expected to return an object containing the computed solution to the optimization
        problem that can be queried.

        Args:
            point (np.ndarray): assignment of parameters for which we are solving the problem.
            fixed_delta (List[bool]): list of boolean values representing the assignment of boolean values to the problem
        """
        raise NotImplementedError()

    @abstractmethod
    def get_output(self, solution) -> List[bool]:
        """
        Retrieves the assignment of boolean values from the solution of the optimization problem.

        Args:
            solution: object containing an assignment of values to the variables of the optimization problem.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_output_size(self) -> int:
        """
        Returns the size of the assignment of boolean variables we want to predict.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_solver(self, theta_domain: Polyhedron, fixed_delta) -> list[tuple[Context, Solver, list[Real]]]:
        """
        Returns a portfolio of Z3 solvers instructed to check whether the given assignment of boolean values for the
        optimization problem is feasible for every point contained in the given polyhedral domain.

        Args:
            theta_domain (Polyhedron): polyhedral domain being verified.
            fixed_delta (List[bool]): assignment of boolean values to the problem beingh verified.
        """
        raise NotImplementedError

    @abstractmethod
    def get_bit_resolution(self) -> int:
        """
        Returns the supposed bit resolution of sensors measuring the variables of the optimization problem. Used to
        define numerical tolerances of the algorithm.
        """
        raise NotImplementedError

    @abstractmethod
    def get_parameter_domain_interval(self) -> Tuple[np.array, np.array]:
        """
        Returns the interval of values that parameter assignments can take.
        """
        raise NotImplementedError

    def get_resolution(self) -> np.ndarray:
        """
        Used to discretize the parameter domain into values that can be measured given the bit resolution.
        """
        min, max = self.get_parameter_domain_interval()
        res = np.zeros(len(min))
        for i in range(len(min)):
            res[i] = (max[i] - min[i]) / (2**self.get_bit_resolution() - 1)
        return res

    def discretize(self, point: np.ndarray):
        """
        Discretizes the given point according to the bit resolution.

        Args:
            point (np.ndarray): assignment of parameters.
        """
        min, max = self.get_parameter_domain_interval()
        res = self.get_resolution()
        disc = np.zeros(len(res))
        for i in range(len(point)):
            disc[i] = min[i] + round((point[i] - min[i]) / res[i]) * res[i]
        return disc

    def same_discretization(self, min, max) -> bool:
        """
        Check whether the minimum and maximum point in a hyper-rectangular domain have the same discretization.

        Args:
            min (np.ndarray): minimum point of the domain.
            max (np.ndarray): maximum point of the domain.
        """
        return np.all(self.discretize(min) == self.discretize(max))