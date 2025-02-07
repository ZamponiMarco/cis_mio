from typing import List

import numpy as np
import pathos.multiprocessing
from system.solvable import Solvable


def construct_big_M(binary, M, lhs, rhs):
    return M * binary >= rhs - lhs, M * (1 - binary) >= lhs - rhs

def solve_function(solvable: Solvable):
    return lambda point: solvable.get_output(solvable.solve(point))

def parallel_solve(solvable: Solvable, points: np.ndarray) -> List[List[bool]]:
    with pathos.multiprocessing.ThreadPool() as pool:
        # Use map to parallelize the work
        results = pool.map(solve_function(solvable), points)
        pool.close()
    return results

def sequential_solve(solvable: Solvable, points: np.ndarray) -> List[List[bool]]:
    return [solve_function(solvable)(point) for point in points]


def box_constraint(model, delta_variables, state_variables, zone):
    lb, ub = zone.bounding_box()
    x_min = construct_big_M(delta_variables[0], 1, state_variables[0], lb[0])
    x_max = construct_big_M(delta_variables[1], 1, -state_variables[0], -ub[0])
    y_min = construct_big_M(delta_variables[2], 1, state_variables[1], lb[1])
    y_max = construct_big_M(delta_variables[3], 1, -state_variables[1], -ub[1])

    model.addConstr(x_min[0])
    model.addConstr(x_min[1])
    model.addConstr(x_max[0])
    model.addConstr(x_max[1])
    model.addConstr(y_min[0])
    model.addConstr(y_min[1])
    model.addConstr(y_max[0])
    model.addConstr(y_max[1])
