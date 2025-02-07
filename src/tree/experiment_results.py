from set.polyhedron import Polyhedron
from typing import List


class ExperimentResult:
    """
    A class to store and represent the results of a certified tree generation experiment.

    Attributes:
        seed (int): the seed for the random number generator.
        execution_time (float): the total execution time of the tree generation.
        max_height (int): the maximum height of the generated tree.
        utilization (float): approximation of CPU usage during tree generation.
        submission_times (List[float]): timestamps of node processing tasks submissions
        completion_times (List[float]): timestamps of node processing tasks completions
        verified_leaves (int): amount of verified leaves in the tree.
        total_leaves (int): amount of total leaves in the tree.
        total_nodes (int): amount of total nodes in the tree.
        verified_volume (float): percentage of volume of the parameter space that is verified.
        verification_zone (Polyhedron): parameter space.
        data_points (int): total amount of data points used to construct the tree.
    """
    def __init__(
            self,
            seed: int,
            execution_time: float,
            max_height: int,
            utilization: float,
            submission_times: List[float],
            completion_times: List[float],
            verified_leaves: int,
            total_leaves: int,
            total_nodes: int,
            verified_volume: float,
            verification_zone: Polyhedron,
            data_points: int
    ) -> None:
        """
        Initialize an ExperimentResult object.
        """
        self.seed = seed
        self.execution_time = execution_time
        self.max_height = max_height
        self.utilization = utilization
        self.submission_times = submission_times
        self.completion_times = completion_times
        self.verified_leaves = verified_leaves
        self.total_leaves = total_leaves
        self.total_nodes = total_nodes
        self.verified_volume = verified_volume
        self.verification_zone = verification_zone
        self.data_points = data_points

    def __str__(self) -> str:
        """
        Return a formatted string representation of the object.
        """
        return (
            f"ExperimentResults(\n"
            f"  Seed: {self.seed},\n"
            f"  Execution Time: {self.execution_time:.2f}s,\n"
            f"  Max Height: {self.max_height},\n"
            f"  Verified Leaves: {self.verified_leaves}/{self.total_leaves},\n"
            f"  Total Nodes: {self.total_nodes},\n"
            f"  Verified Volume: {self.verified_volume/self.verification_zone.volume()},\n"
            f"  Data Points: {self.data_points}\n"
            f")"
        )
