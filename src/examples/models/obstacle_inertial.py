import math
from typing import List, Tuple

import gurobipy
import numpy as np
from gurobipy import GRB
from set.polyhedron import Polyhedron, from_interval
from system.solvable import Solvable
from util.optimization_util import box_constraint
from z3 import Solver, With, Then, Real, RealVector, And, simplify, Or, Not, ForAll, BoolVal, \
    RealVal, Tactic, Context, If, BoolVector, sat, unsat

PROBLEM_FILE_MIP = 'resources/deadzone/problem_mip.mps'
PROBLEM_FILE_P = 'resources/deadzone/problem_p.mps'

VELOCITY_UPPER = 0.05

ACCEL_UPPER = 0.05

POINT_LOWER = 0.0
POINT_UPPER = 1.0

TIME_STEP = 0.2

class ObstacleInertial(Solvable):

    def __init__(self, horizon, obstacles: List[Polyhedron]):
        self.horizon = horizon
        self.obstacles = obstacles

    def solve(self, point):
        with gurobipy.Env(params={'OutputFlag':0}) as env:
            m = gurobipy.Model(env=env)
            self.get_model(m)
            m.getVarByName(f'x[0]').lb = point[0]
            m.getVarByName(f'x[0]').ub = point[0]
            m.getVarByName(f'y[0]').lb = point[1]
            m.getVarByName(f'y[0]').ub = point[1]
            m.getVarByName(f'v_x[0]').lb = point[2]
            m.getVarByName(f'v_x[0]').ub = point[2]
            m.getVarByName(f'v_y[0]').lb = point[3]
            m.getVarByName(f'v_y[0]').ub = point[3]

            m.Params.TimeLimit = 60

            m.update()
            m.optimize()
            return m

    def solve_fixed(self, point, fixed_delta):
        with gurobipy.Env(params={'OutputFlag':0}) as env:
            m = gurobipy.Model(env=env)
            self.get_model(m)
            m.getVarByName(f'x[0]').lb = point[0]
            m.getVarByName(f'x[0]').ub = point[0]
            m.getVarByName(f'y[0]').lb = point[1]
            m.getVarByName(f'y[0]').ub = point[1]
            m.getVarByName(f'v_x[0]').lb = point[2]
            m.getVarByName(f'v_x[0]').ub = point[2]
            m.getVarByName(f'v_y[0]').lb = point[3]
            m.getVarByName(f'v_y[0]').ub = point[3]
            for step in range(self.horizon):
                m.getVarByName(f'delta_obs_0_0[{step + 1}]').lb = fixed_delta[step]
                m.getVarByName(f'delta_obs_0_0[{step + 1}]').ub = fixed_delta[step]
                m.getVarByName(f'delta_obs_0_1[{step + 1}]').lb = fixed_delta[self.horizon + step]
                m.getVarByName(f'delta_obs_0_1[{step + 1}]').ub = fixed_delta[self.horizon + step]
                m.getVarByName(f'delta_obs_0_2[{step + 1}]').lb = fixed_delta[2 * self.horizon + step]
                m.getVarByName(f'delta_obs_0_2[{step + 1}]').ub = fixed_delta[2 * self.horizon + step]
                m.getVarByName(f'delta_obs_0_3[{step + 1}]').lb = fixed_delta[3 * self.horizon + step]
                m.getVarByName(f'delta_obs_0_3[{step + 1}]').ub = fixed_delta[3 * self.horizon + step]

            lb, ub = self.obstacles[0].bounding_box()
            m.getVarByName(f'delta_obs_0_0[0]').lb = point[0] <= lb[0]
            m.getVarByName(f'delta_obs_0_0[0]').ub = point[0] <= lb[0]
            m.getVarByName(f'delta_obs_0_1[0]').lb = point[0] >= ub[0]
            m.getVarByName(f'delta_obs_0_1[0]').ub = point[0] >= ub[0]
            m.getVarByName(f'delta_obs_0_2[0]').lb = point[1] <= lb[1]
            m.getVarByName(f'delta_obs_0_2[0]').ub = point[1] <= lb[1]
            m.getVarByName(f'delta_obs_0_3[0]').lb = point[1] >= ub[1]
            m.getVarByName(f'delta_obs_0_3[0]').ub = point[1] >= ub[1]

            m.Params.TimeLimit = 60

            m.update()
            m.optimize()
            return m

    def get_output(self, solution):
        status = solution.Status

        if status == 2 or status == 13:  # Ok or Timeout with solution
            output = []
            for step in range(1, self.horizon + 1):
                output += [round(solution.getVarByName(f'delta_obs_0_0[{step}]').X)]
            for step in range(1, self.horizon + 1):
                output += [round(solution.getVarByName(f'delta_obs_0_1[{step}]').X)]
            for step in range(1, self.horizon + 1):
                output += [round(solution.getVarByName(f'delta_obs_0_2[{step}]').X)]
            for step in range(1, self.horizon + 1):
                output += [round(solution.getVarByName(f'delta_obs_0_3[{step}]').X)]
            return output
        else:
            print(status, solution.getVarByName('x[0]').lb, solution.getVarByName('y[0]').lb,
                  solution.getVarByName('v_x[0]').lb, solution.getVarByName('v_y[0]').lb)

        return None

    def get_output_size(self):
        return 4 * self.horizon

    def get_model(self, model) -> None:

        # State variables
        x = model.addVars(self.horizon + 1, name='x', lb=-math.inf)
        y = model.addVars(self.horizon + 1, name='y', lb=-math.inf)
        v_x = model.addVars(self.horizon + 1, name='v_x', lb=-math.inf)
        v_y = model.addVars(self.horizon + 1, name='v_y', lb=-math.inf)

        # Control inputs (accelerations)
        a_x = model.addVars(self.horizon, name='a_x', lb=-math.inf)
        a_y = model.addVars(self.horizon, name='a_y', lb=-math.inf)

        # Dynamics
        for step in range(self.horizon):
            # Position update based on velocity
            model.addConstr(x[step + 1] == x[step] + TIME_STEP * v_x[step])
            model.addConstr(y[step + 1] == y[step] + TIME_STEP * v_y[step])

            # Velocity update based on acceleration
            model.addConstr(v_x[step + 1] == v_x[step] + TIME_STEP * a_x[step])
            model.addConstr(v_y[step + 1] == v_y[step] + TIME_STEP * a_y[step])

        # State Bounds
        for step in range(self.horizon + 1):
            # Position bounds
            model.addConstr(x[step] >= POINT_LOWER)
            model.addConstr(x[step] <= POINT_UPPER)
            model.addConstr(y[step] >= POINT_LOWER)
            model.addConstr(y[step] <= POINT_UPPER)

            # Velocity bounds
            model.addConstr(v_x[step] <= VELOCITY_UPPER)
            model.addConstr(v_x[step] >= -VELOCITY_UPPER)
            model.addConstr(v_y[step] <= VELOCITY_UPPER)
            model.addConstr(v_y[step] >= -VELOCITY_UPPER)

        # Input Bounds and Constraints on Accelerations
        for step in range(self.horizon):
            # Acceleration magnitude upper bound
            model.addConstr(a_x[step] <= ACCEL_UPPER)
            model.addConstr(a_x[step] >= -ACCEL_UPPER)
            model.addConstr(a_y[step] <= ACCEL_UPPER)
            model.addConstr(a_y[step] >= -ACCEL_UPPER)


        # Obstacle Constraints
        delta_obs_0_0 = model.addVars(self.horizon + 1, name='delta_obs_0_0', vtype=GRB.BINARY)
        delta_obs_0_1 = model.addVars(self.horizon + 1, name='delta_obs_0_1', vtype=GRB.BINARY)
        delta_obs_0_2 = model.addVars(self.horizon + 1, name='delta_obs_0_2', vtype=GRB.BINARY)
        delta_obs_0_3 = model.addVars(self.horizon + 1, name='delta_obs_0_3', vtype=GRB.BINARY)

        for step in range(self.horizon + 1):
            box_constraint(
                model,
                [delta_obs_0_0[step], delta_obs_0_1[step], delta_obs_0_2[step], delta_obs_0_3[step]],
                [x[step], y[step]],
                self.obstacles[0]
            )
            model.addConstr(delta_obs_0_0[step] + delta_obs_0_1[step] +
                            delta_obs_0_2[step] + delta_obs_0_3[step] >= 1)

        # Objective Function
        model.setObjective(
            sum(
                (1 / 10) * x[step] ** 2 + (1 / 10) * y[step] ** 2 +
                (1 / 100) * v_x[step] ** 2 + (1 / 100) * v_y[step] ** 2 +
                (1 / 1000) * a_x[step] ** 2 + (1 / 1000) * a_y[step] ** 2
                for step in range(self.horizon)
            ) +
            x[self.horizon] ** 2 + y[self.horizon] ** 2 +
            v_x[self.horizon] ** 2 + v_y[self.horizon] ** 2,
            GRB.MINIMIZE
        )

        model.update()

    def get_solver(self, theta_domain: Polyhedron, fixed_delta) -> list[tuple[Context, Solver, list[Real]]]:
        context_1 = Context()
        solver_1 = Then(
            With(Tactic('simplify', ctx=context_1), som=True, sort_sums=True),
            With(Tactic('qe', ctx=context_1)),
            Tactic('purify-arith', ctx=context_1),
            Tactic('smt', ctx=context_1)
        ).solver()
        x_initial_1, y_initial_1, v_x_initial1, v_y_initial1 = self._build_context(context_1, solver_1, theta_domain, fixed_delta)

        context_2 = Context()
        solver_2 = Then(
            With(Tactic('simplify', ctx=context_2), som=True, sort_sums=True),
            Tactic('smt', ctx=context_2)
        ).solver()
        x_initial_2, y_initial_2, v_x_initial2, v_y_initial2 = self._build_context(context_2, solver_2, theta_domain, fixed_delta)

        context_3 = Context()
        solver_3 = Tactic('smt', ctx=context_3).solver()
        x_initial_3, y_initial_3, v_x_initial3, v_y_initial3 = self._build_context(context_3, solver_3, theta_domain, fixed_delta)

        return [
            (context_1, solver_1, [x_initial_1, y_initial_1, v_x_initial1, v_y_initial1]),
            (context_2, solver_2, [x_initial_2, y_initial_2, v_x_initial2, v_y_initial2]),
            (context_3, solver_3, [x_initial_3, y_initial_3, v_x_initial3, v_y_initial3])
        ]

    def get_bit_resolution(self) -> int:
        return 10

    def get_parameter_domain_interval(self) -> Tuple[np.array, np.array]:
        return np.array([POINT_LOWER, POINT_LOWER, -VELOCITY_UPPER, -VELOCITY_UPPER]), \
               np.array([POINT_UPPER, POINT_UPPER, VELOCITY_UPPER, VELOCITY_UPPER])

    def _build_context(self, context, solver, theta_domain, fixed_delta):
        epsilon = RealVal(self.get_resolution()[0]/2, ctx=context)

        # Initial positions and velocities
        x_initial = Real('x_initial', ctx=context)
        y_initial = Real('y_initial', ctx=context)
        v_x_initial = Real('v_x_initial', ctx=context)
        v_y_initial = Real('v_y_initial', ctx=context)

        # State variables (positions and velocities)
        x = [x_initial] + RealVector('x', self.horizon, ctx=context)
        y = [y_initial] + RealVector('y', self.horizon, ctx=context)
        v_x = [v_x_initial] + RealVector('v_x', self.horizon, ctx=context)
        v_y = [v_y_initial] + RealVector('v_y', self.horizon, ctx=context)

        # Control inputs (accelerations)
        a_x = RealVector('a_x', self.horizon, ctx=context)
        a_y = RealVector('a_y', self.horizon, ctx=context)

        fixed_delta = [BoolVal(bool(el), ctx=context) for el in fixed_delta]

        # Define delta_u and obstacle delta variables using fixed_delta
        lb, ub = self.obstacles[0].bounding_box()
        delta_0_0 = [x_initial <= RealVal(lb[0], ctx=context)] + \
                    fixed_delta[0:self.horizon]
        delta_0_1 = [x_initial >= RealVal(ub[0], ctx=context)] + \
                    fixed_delta[self.horizon:2 * self.horizon]
        delta_0_2 = [y_initial <= RealVal(lb[1], ctx=context)] + \
                    fixed_delta[2 * self.horizon:3 * self.horizon]
        delta_0_3 = [y_initial >= RealVal(ub[1], ctx=context)] + \
                    fixed_delta[3 * self.horizon:4 * self.horizon]

        # Build dynamics constraints
        dyn = self._build_dyn(x, y, v_x, v_y, a_x, a_y)

        # Build feasibility constraints
        feas = self._build_feas(
            epsilon, x, y, v_x, v_y, a_x, a_y,
            delta_0_0, delta_0_1, delta_0_2, delta_0_3, context)

        # Build domain constraints (include initial positions and velocities)
        dom = self._build_dom(theta_domain, x_initial, y_initial, v_x_initial, v_y_initial)

        # Define the formula to check
        to_check = simplify(ForAll(
            x[1:] + y[1:] + v_x[1:] + v_y[1:] + a_x + a_y,
            Not(And(dyn, feas))))

        # Add constraints to the solver
        solver.add(dom)
        solver.add(to_check)

        # Return initial positions and velocities
        return x_initial, y_initial, v_x_initial, v_y_initial

    def _build_dom(self, zone: Polyhedron, x_initial, y_initial, v_x_initial, v_y_initial):
        dom = zone.contains_symbolic([x_initial, y_initial, v_x_initial, v_y_initial])
        return simplify(dom)

    def _build_dyn(self, x, y, v_x, v_y, a_x, a_y):
        dyn_constraints = []
        for step in range(self.horizon):
            # Position updates
            dyn_constraints.append(x[step + 1] == x[step] + TIME_STEP * v_x[step])
            dyn_constraints.append(y[step + 1] == y[step] + TIME_STEP * v_y[step])

            # Velocity updates
            dyn_constraints.append(v_x[step + 1] == v_x[step] + TIME_STEP * a_x[step])
            dyn_constraints.append(v_y[step + 1] == v_y[step] + TIME_STEP * a_y[step])
        return simplify(And(dyn_constraints))

    def _build_feas(self, epsilon, x, y, v_x, v_y, a_x, a_y,
                    delta_0_0, delta_0_1, delta_0_2, delta_0_3, context=None):
        feas_constraints = []

        # State Bounds
        for step in range(1, self.horizon + 1):
            feas_constraints.extend([
                x[step] >= POINT_LOWER, x[step] <= POINT_UPPER,
                y[step] >= POINT_LOWER, y[step] <= POINT_UPPER,
                v_x[step] >= -VELOCITY_UPPER, v_x[step] <= VELOCITY_UPPER,
                v_y[step] >= -VELOCITY_UPPER, v_y[step] <= VELOCITY_UPPER
            ])

        # Control Input Bounds
        for step in range(self.horizon):
            feas_constraints.extend([
                a_x[step] >= -ACCEL_UPPER, a_x[step] <= ACCEL_UPPER,
                a_y[step] >= -ACCEL_UPPER, a_y[step] <= ACCEL_UPPER
            ])

        # Obstacle Avoidance Constraints
        lb, ub = self.obstacles[0].bounding_box()
        lbx = RealVal(lb[0], context)
        ubx = RealVal(ub[0], context)
        lby = RealVal(lb[1], context)
        uby = RealVal(ub[1], context)

        for step in range(self.horizon + 1):
            feas_constraints.append(
                Or(delta_0_0[step], delta_0_1[step], delta_0_2[step], delta_0_3[step]))

            feas_constraints.append(
                If(delta_0_0[step], x[step] <= lbx + epsilon, x[step] > lbx - epsilon, ctx=context))
            feas_constraints.append(
                If(delta_0_1[step], x[step] >= ubx - epsilon, x[step] < ubx + epsilon, ctx=context))
            feas_constraints.append(
                If(delta_0_2[step], y[step] <= lby + epsilon, y[step] > lby - epsilon, ctx=context))
            feas_constraints.append(
                If(delta_0_3[step], y[step] >= uby - epsilon, y[step] < uby + epsilon, ctx=context))

        return simplify(And(feas_constraints))

    def test(self, theta_domain):
        solver = Solver()

        epsilon = RealVal(0)

        # Initial positions and velocities
        x_initial = Real('x_initial')
        y_initial = Real('y_initial')
        v_x_initial = Real('v_x_initial')
        v_y_initial = Real('v_y_initial')

        # State variables (positions and velocities)
        x = [x_initial] + RealVector('x', self.horizon)
        y = [y_initial] + RealVector('y', self.horizon)
        v_x = [v_x_initial] + RealVector('v_x', self.horizon)
        v_y = [v_y_initial] + RealVector('v_y', self.horizon)

        # Control inputs (accelerations)
        a_x = RealVector('a_x', self.horizon)
        a_y = RealVector('a_y', self.horizon)

        # Define obstacle delta variables using fixed_delta
        lb, ub = self.obstacles[0].bounding_box()
        delta_0_0 = [x_initial <= RealVal(lb[0])] + \
                    BoolVector('delta_0_0', self.horizon)
        delta_0_1 = [x_initial >= RealVal(ub[0])] + \
                    BoolVector('delta_0_1', self.horizon)
        delta_0_2 = [y_initial <= RealVal(lb[1])] + \
                    BoolVector('delta_0_2', self.horizon)
        delta_0_3 = [y_initial >= RealVal(ub[1])] + \
                    BoolVector('delta_0_3', self.horizon)

        # Build dynamics constraints
        dyn = self._build_dyn(x, y, v_x, v_y, a_x, a_y)

        # Build feasibility constraints
        feas = self._build_feas(
            epsilon, x, y, v_x, v_y, a_x, a_y,
            delta_0_0, delta_0_1, delta_0_2, delta_0_3)

        # Build domain constraints (include initial positions and velocities)
        dom = self._build_dom(theta_domain, x_initial, y_initial, v_x_initial, v_y_initial)

        # Define the formula to check
        to_check = ForAll(
            x[1:] + y[1:] + v_x[1:] + v_y[1:] + a_x + a_y + delta_0_0[1:] + delta_0_1[1:] + delta_0_2[1:] + delta_0_3[1:],
            Not(And(dyn, feas)))

        # Add constraints to the solver
        solver.add(dom)
        solver.add(to_check)

        print(solver.check())
        if solver.check() == sat:
            print(solver.model())
        elif solver.check() == unsat:
            print(solver.unsat_core())

def example_obstacle_inertial() -> ObstacleInertial:
    return ObstacleInertial(
        5,
        [
            from_interval(np.array([0.1, 0.4]), np.array([0.7, 0.5])),
        ]
    )