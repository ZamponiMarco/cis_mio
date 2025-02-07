import cvxpy as cp
import cvxpy.settings as s
import numpy as np
import osqp
import scipy.sparse as sp

from set.polyhedron import from_interval
from util.optimization_util import construct_big_M

VELOCITY_UPPER = 0.05
ACCEL_UPPER = 0.05
POINT_LOWER = 0.0
POINT_UPPER = 1.0
TIME_STEP = 0.2
HORIZON = 5

obstacles = [
    from_interval(np.array([0.1, 0.4]), np.array([0.7, 0.5])),
]


def solve_mpc_osqp(initial_point, fixed_delta):
    # Constants
    horizon = HORIZON

    # State variables
    initial_x = cp.Parameter()  # Define the starting parameter for x
    initial_y = cp.Parameter()  # Define the starting parameter for y
    initial_v_x = cp.Parameter()  # Define the starting parameter for v_x
    initial_v_y = cp.Parameter()  # Define the starting parameter for v_y

    x = [initial_x] + [cp.Variable() for _ in range(horizon)]
    y = [initial_y] + [cp.Variable() for _ in range(horizon)]
    v_x = [initial_v_x] + [cp.Variable() for _ in range(horizon)]
    v_y = [initial_v_y] + [cp.Variable() for _ in range(horizon)]

    # Control inputs (accelerations)
    a_x = [cp.Variable() for _ in range(horizon)]  # 5
    a_y = [cp.Variable() for _ in range(horizon)]  # 5

    delta_obs_0_0 = [cp.Parameter() for _ in range(horizon + 1)]  # 6
    delta_obs_0_1 = [cp.Parameter() for _ in range(horizon + 1)]  # 6
    delta_obs_0_2 = [cp.Parameter() for _ in range(horizon + 1)]  # 6
    delta_obs_0_3 = [cp.Parameter() for _ in range(horizon + 1)]  # 6

    # Constraints list
    constraints = []

    # Dynamics constraints
    for step in range(horizon):
        # Position update based on velocity
        constraints.append(x[step + 1] == x[step] + TIME_STEP * v_x[step])
        constraints.append(y[step + 1] == y[step] + TIME_STEP * v_y[step])

        # Velocity update based on acceleration
        constraints.append(v_x[step + 1] == v_x[step] + TIME_STEP * a_x[step])
        constraints.append(v_y[step + 1] == v_y[step] + TIME_STEP * a_y[step])

    # State bounds
    for step in range(horizon + 1):
        # Position bounds
        constraints.append(x[step] >= POINT_LOWER)
        constraints.append(x[step] <= POINT_UPPER)
        constraints.append(y[step] >= POINT_LOWER)
        constraints.append(y[step] <= POINT_UPPER)

        # Velocity bounds
        constraints.append(v_x[step] <= VELOCITY_UPPER)
        constraints.append(v_x[step] >= -VELOCITY_UPPER)
        constraints.append(v_y[step] <= VELOCITY_UPPER)
        constraints.append(v_y[step] >= -VELOCITY_UPPER)

    # Input bounds and constraints on accelerations
    for step in range(horizon):
        # Acceleration magnitude upper bound
        constraints.append(a_x[step] <= ACCEL_UPPER)
        constraints.append(a_x[step] >= -ACCEL_UPPER)
        constraints.append(a_y[step] <= ACCEL_UPPER)
        constraints.append(a_y[step] >= -ACCEL_UPPER)

    for step in range(horizon + 1):
        # Assuming box_constraint can be adapted to CVXPY (custom function needed)
        lb, ub = obstacles[0].bounding_box()
        x_min = construct_big_M(delta_obs_0_0[step], 1, x[step], lb[0])
        x_max = construct_big_M(delta_obs_0_1[step], 1, -x[step], -ub[0])
        y_min = construct_big_M(delta_obs_0_2[step], 1, y[step], lb[1])
        y_max = construct_big_M(delta_obs_0_3[step], 1, -y[step], -ub[1])

        constraints.append(x_min[0])
        constraints.append(x_min[1])
        constraints.append(x_max[0])
        constraints.append(x_max[1])
        constraints.append(y_min[0])
        constraints.append(y_min[1])
        constraints.append(y_max[0])
        constraints.append(y_max[1])

        constraints.append(delta_obs_0_0[step] + delta_obs_0_1[step] +
                           delta_obs_0_2[step] + delta_obs_0_3[step] >= 1)

    # Objective function
    objective = cp.Minimize(
        sum(
            (1 / 10) * cp.square(x[step]) + (1 / 10) * cp.square(y[step]) +
            (1 / 100) * cp.square(v_x[step]) + (1 / 100) * cp.square(v_y[step]) +
            (1 / 1000) * cp.square(a_x[step]) + (1 / 1000) * cp.square(a_y[step])
            for step in range(horizon)
        ) +
        cp.square(x[horizon]) + cp.square(y[horizon]) +
        cp.square(v_x[horizon]) + cp.square(v_y[horizon])
    )

    problem = cp.Problem(objective, constraints)

    # Set initial point
    x[0].value = initial_point[0]
    y[0].value = initial_point[1]
    v_x[0].value = initial_point[2]
    v_y[0].value = initial_point[3]

    delta_obs_0_0[0].value = (1 if initial_point[0] <= 0.1 else 0)
    delta_obs_0_1[0].value = (1 if initial_point[0] >= 0.7 else 0)
    delta_obs_0_2[0].value = (1 if initial_point[1] <= 0.4 else 0)
    delta_obs_0_3[0].value = (1 if initial_point[1] >= 0.5 else 0)

    for step in range(HORIZON):
        delta_obs_0_0[step + 1].value = fixed_delta[step]
        delta_obs_0_1[step + 1].value = fixed_delta[HORIZON + step]
        delta_obs_0_2[step + 1].value = fixed_delta[2 * HORIZON + step]
        delta_obs_0_3[step + 1].value = fixed_delta[3 * HORIZON + step]

    data = problem.get_problem_data(cp.OSQP)[0]

    P = data[s.P]
    q = data[s.Q]
    A = sp.vstack([data[s.A], data[s.F]]).tocsc()
    uA = np.concatenate((data[s.B], data[s.G]))
    lA = np.concatenate([data[s.B], -np.inf * np.ones(data[s.G].shape)])

    osqp_settings = {
        'eps_abs': 1e-4,
        'eps_rel': 1e-4,
        'eps_prim_inf': 1/2048,
        'verbose': False
    }

    solver = osqp.OSQP()
    solver.setup(P, q, A, lA, uA, **osqp_settings)
    results = solver.solve()

    return results

if __name__ == "__main__":
    # Example usage
    initial_point = [0.11, 0.11, 0, 0]
    fixed_delta = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    results = solve_mpc_osqp(initial_point, fixed_delta)
    print(results.x)
    print('x:', results.x[4:5*HORIZON + 5:6])
    print('y:', results.x[5:5*HORIZON + 6:6])
    print('v_x:', results.x[2:5*HORIZON + 3:6])
    print('v_y:', results.x[3:5*HORIZON + 4:6])
    print('a_x:', results.x[0:5*HORIZON + 1:6])
    print('a_y:', results.x[1:5*HORIZON + 2:6])
    print(results.info.run_time)
    print(results.info.status)
    print(results.info.obj_val)
