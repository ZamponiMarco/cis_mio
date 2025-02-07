import math
from typing import List, Tuple

import gurobipy
import numpy as np
from gurobipy import GRB
from set.polyhedron import Polyhedron, from_interval
from system.solvable import Solvable
from util.optimization_util import construct_big_M, box_constraint
from z3 import Solver, With, Then, Real, RealVector, And, simplify, If, Or, Not, ForAll, BoolVal, \
    RealVal, Tactic, Context

PROBLEM_FILE_MIP = 'resources/deadzone/problem_mip.mps'
PROBLEM_FILE_P = 'resources/deadzone/problem_p.mps'

SPEED_LOWER = 0.03
SPEED_UPPER = 0.08

SPEED_LOWER_SQUARED = SPEED_LOWER ** 2
SPEED_UPPER_SQUARED = SPEED_UPPER ** 2

POINT_LOWER = 0
POINT_UPPER = 1

TIME_STEP = 1


class Obstacle(Solvable):

    def __init__(self, horizon, obstacles: List[Polyhedron]):
        self.horizon = horizon
        self.obstacles = obstacles

    def solve(self, point):
        with gurobipy.Env(params={'OutputFlag': 0}) as env:
            m = gurobipy.Model(env=env)
            self.get_model(m)
            m.getVarByName(f'x[0]').lb = point[0]
            m.getVarByName(f'x[0]').ub = point[0]
            m.getVarByName(f'y[0]').lb = point[1]
            m.getVarByName(f'y[0]').ub = point[1]
            m.update()
            m.optimize()
            return m

    def solve_fixed(self, point, fixed_delta):
        with gurobipy.Env(params={'OutputFlag': 0}) as env:
            m = gurobipy.Model(env=env)
            self.get_model(m)
            m.getVarByName(f'x[0]').lb = point[0]
            m.getVarByName(f'x[0]').ub = point[0]
            m.getVarByName(f'y[0]').lb = point[1]
            m.getVarByName(f'y[0]').ub = point[1]
            for step in range(self.horizon):
                m.getVarByName(f'delta_u[{step}]').lb = fixed_delta[step]
                m.getVarByName(f'delta_u[{step}]').ub = fixed_delta[step]
                m.getVarByName(f'delta_obs_0_0[{step + 1}]').lb = fixed_delta[self.horizon + step]
                m.getVarByName(f'delta_obs_0_0[{step + 1}]').ub = fixed_delta[self.horizon + step]
                m.getVarByName(f'delta_obs_0_1[{step + 1}]').lb = fixed_delta[2 * self.horizon + step]
                m.getVarByName(f'delta_obs_0_1[{step + 1}]').ub = fixed_delta[2 * self.horizon + step]
                m.getVarByName(f'delta_obs_0_2[{step + 1}]').lb = fixed_delta[3 * self.horizon + step]
                m.getVarByName(f'delta_obs_0_2[{step + 1}]').ub = fixed_delta[3 * self.horizon + step]
                m.getVarByName(f'delta_obs_0_3[{step + 1}]').lb = fixed_delta[4 * self.horizon + step]
                m.getVarByName(f'delta_obs_0_3[{step + 1}]').ub = fixed_delta[4 * self.horizon + step]

            lb, ub = self.obstacles[0].bounding_box()
            m.getVarByName(f'delta_obs_0_0[0]').lb = point[0] <= lb[0]
            m.getVarByName(f'delta_obs_0_0[0]').ub = point[0] <= lb[0]
            m.getVarByName(f'delta_obs_0_1[0]').lb = point[0] >= ub[0]
            m.getVarByName(f'delta_obs_0_1[0]').ub = point[0] >= ub[0]
            m.getVarByName(f'delta_obs_0_2[0]').lb = point[1] <= lb[1]
            m.getVarByName(f'delta_obs_0_2[0]').ub = point[1] <= lb[1]
            m.getVarByName(f'delta_obs_0_3[0]').lb = point[1] >= ub[1]
            m.getVarByName(f'delta_obs_0_3[0]').ub = point[1] >= ub[1]

            m.update()
            m.optimize()
            return m

    def get_output(self, solution):
        status = solution.Status

        if status == 2 or status == 13:  # Ok or Timeout with solution
            output = []
            for step in range(self.horizon):
                output += [round(solution.getVarByName(f'delta_u[{step}]').X)]
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
            return None

    def get_output_size(self):
        return 5 * self.horizon

    def get_model(self, model) -> None:

        x = model.addVars(self.horizon + 1, name='x', lb=-math.inf)
        y = model.addVars(self.horizon + 1, name='y', lb=-math.inf)

        u_x = model.addVars(self.horizon, name='u_x', lb=-math.inf)
        u_y = model.addVars(self.horizon, name='u_y', lb=-math.inf)

        delta_u = model.addVars(self.horizon, name='delta_u', vtype=GRB.BINARY)

        z_x = model.addVars(self.horizon, name='z_x', lb=-math.inf)
        z_y = model.addVars(self.horizon, name='z_y', lb=-math.inf)

        # Dynamics
        for step in range(self.horizon):
            model.addConstr(x[step + 1] == x[step] + TIME_STEP * (u_x[step] - z_x[step]))
            model.addConstr(y[step + 1] == y[step] + TIME_STEP * (u_y[step] - z_y[step]))

        # State Bounds
        for step in range(self.horizon + 1):
            model.addConstr(x[step] >= POINT_LOWER)
            model.addConstr(x[step] <= POINT_UPPER)
            model.addConstr(y[step] >= POINT_LOWER)
            model.addConstr(y[step] <= POINT_UPPER)

        # Input Bounds
        for step in range(self.horizon):
            delta_u_big_M = construct_big_M(delta_u[step], 1, u_x[step] ** 2 + u_y[step] ** 2, SPEED_LOWER_SQUARED)

            model.addConstr(delta_u_big_M[0])
            model.addConstr(delta_u_big_M[1])

            model.addConstr(u_x[step] ** 2 + u_y[step] ** 2 <= SPEED_UPPER_SQUARED)

            model.addConstr(z_x[step] <= 1 * delta_u[step])
            model.addConstr(z_x[step] >= -1 * delta_u[step])
            model.addConstr(z_x[step] - u_x[step] <= 1 * (1 - delta_u[step]))
            model.addConstr(z_x[step] - u_x[step] >= -1 * (1 - delta_u[step]))

            model.addConstr(z_y[step] <= 1 * delta_u[step])
            model.addConstr(z_y[step] >= -1 * delta_u[step])
            model.addConstr(z_y[step] - u_y[step] <= 1 * (1 - delta_u[step]))
            model.addConstr(z_y[step] - u_y[step] >= -1 * (1 - delta_u[step]))

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

        model.setObjective(sum(
            1 / 10 * x[step] ** 2 + 1 / 10 * y[step] ** 2 +
            1 / 100 * u_x[step] ** 2 + 1 / 100 * u_y[step] ** 2
            for step in range(self.horizon)) +
                           x[self.horizon] ** 2 +
                           y[self.horizon] ** 2,
                           GRB.MINIMIZE)

        model.update()

    def get_solver(self, theta_domain: Polyhedron, fixed_delta) -> list[tuple[Context, Solver, list[Real]]]:
        context_1 = Context()
        solver_1 = Then(
            With(Tactic('simplify', ctx=context_1), som=True, sort_sums=True),
            With(Tactic('qe', ctx=context_1), qe_nonlinear=True),
            Tactic('purify-arith', ctx=context_1),
            Tactic('smt', ctx=context_1)
        ).solver()
        x_initial_1, y_initial_1 = self._build_context(context_1, solver_1, theta_domain, fixed_delta)

        context_2 = Context()
        solver_2 = Then(
            With(Tactic('simplify', ctx=context_2), som=True, sort_sums=True),
            Tactic('smt', ctx=context_2)
        ).solver()
        x_initial_2, y_initial_2 = self._build_context(context_2, solver_2, theta_domain, fixed_delta)

        context_3 = Context()
        solver_3 = Tactic('smt', ctx=context_3).solver()
        x_initial_3, y_initial_3 = self._build_context(context_3, solver_3, theta_domain, fixed_delta)

        return [
            (context_1, solver_1, [x_initial_1, y_initial_1]),
            (context_2, solver_2, [x_initial_2, y_initial_2]),
            (context_3, solver_3, [x_initial_3, y_initial_3])
        ]

    def get_bit_resolution(self) -> int:
        return 10

    def get_parameter_domain_interval(self) -> Tuple[np.array, np.array]:
        return np.array([POINT_LOWER, POINT_LOWER]), np.array([POINT_UPPER, POINT_UPPER])

    def _build_context(self, context, solver, theta_domain, fixed_delta):
        epsilon = RealVal(self.get_resolution()[0]/2, ctx=context)

        x_initial = Real('x_initial', ctx=context)
        y_initial = Real('y_initial', ctx=context)
        x = [x_initial] + RealVector('x', self.horizon, ctx=context)
        y = [y_initial] + RealVector('y', self.horizon, ctx=context)
        u_x = RealVector('u_x', self.horizon, ctx=context)
        u_y = RealVector('u_y', self.horizon, ctx=context)

        fixed_delta = [BoolVal(bool(el), ctx=context) for el in fixed_delta]

        delta_u = fixed_delta[0:self.horizon]

        lb, ub = self.obstacles[0].bounding_box()
        delta_0_0 = [x_initial <= RealVal(lb[0], ctx=context)] + \
                    fixed_delta[self.horizon:2 * self.horizon]
        delta_0_1 = [x_initial >= RealVal(ub[0], ctx=context)] + \
                    fixed_delta[2 * self.horizon:3 * self.horizon]
        delta_0_2 = [y_initial <= RealVal(lb[1], ctx=context)] + \
                    fixed_delta[3 * self.horizon:4 * self.horizon]
        delta_0_3 = [y_initial >= RealVal(ub[1], ctx=context)] + \
                    fixed_delta[4 * self.horizon:5 * self.horizon]
        dyn = self._build_dyn(x, y, u_x, u_y, delta_u, context)
        feas = self._build_feas(epsilon, x, y, u_x, u_y, delta_0_0, delta_0_1, delta_0_2, delta_0_3, delta_u, context)
        dom = self._build_dom(theta_domain, x_initial, y_initial)
        to_check = simplify(ForAll(
            x[1:] + y[1:] + u_x + u_y,
            Not(And(dyn, feas))))

        solver.add(dom)
        solver.add(
            to_check
        )
        return x_initial, y_initial

    def _build_dom(self, zone, x_initial, y_initial):
        return simplify(zone.contains_symbolic([x_initial, y_initial]))

    def _build_dyn(self, x, y, u_x, u_y, delta_u, context):
        dyn_constraints = []

        for step in range(self.horizon):
            dyn_constraints.extend([
                x[step + 1] == x[step] + If(delta_u[step], 0.0, u_x[step], ctx=context),
                y[step + 1] == y[step] + If(delta_u[step], 0.0, u_y[step], ctx=context)
            ])

        return simplify(And(dyn_constraints))

    def _build_feas(self, epsilon, x, y, u_x, u_y, delta_0_0, delta_0_1, delta_0_2, delta_0_3, delta_u, context):
        feas_constraints = []

        point_lower = RealVal(POINT_LOWER, ctx=context)
        point_upper = RealVal(POINT_UPPER, ctx=context)

        # State Bounds
        for step in range(1, self.horizon + 1):
            feas_constraints.extend(
                [
                    x[step] >= point_lower, x[step] <= point_upper,
                    y[step] >= point_lower, y[step] <= point_upper
                ]
            )

        # Input Bounds
        for step in range(self.horizon):
            speed_lower_squared = RealVal(SPEED_LOWER_SQUARED, ctx=context)
            speed_upper_squared = RealVal(SPEED_UPPER_SQUARED, ctx=context)

            feas_constraints.extend([
                u_x[step] ** 2 + u_y[step] ** 2 <= speed_upper_squared,
            ])

            if not delta_u[step]:
                feas_constraints.extend([
                    u_x[step] ** 2 + u_y[step] ** 2 >= speed_lower_squared
                ])

        # Obstacle Constraints
        lb, ub = self.obstacles[0].bounding_box()
        lbx = RealVal(lb[0], context)
        ubx = RealVal(ub[0], context)
        lby = RealVal(lb[1], context)
        uby = RealVal(ub[1], context)

        for step in range(1, self.horizon + 1):
            if delta_0_0[step]:
                feas_constraints.append(x[step] <= lbx + epsilon)
            else:
                feas_constraints.append(x[step] >= lbx - epsilon)
            if delta_0_1[step]:
                feas_constraints.append(x[step] >= ubx - epsilon)
            else:
                feas_constraints.append(x[step] <= ubx + epsilon)
            if delta_0_2[step]:
                feas_constraints.append(y[step] <= lby + epsilon)
            else:
                feas_constraints.append(y[step] >= lby - epsilon)
            if delta_0_3[step]:
                feas_constraints.append(y[step] >= uby - epsilon)
            else:
                feas_constraints.append(y[step] <= uby + epsilon)

            feas_constraints.append(
                Or(delta_0_0[step], delta_0_1[step], delta_0_2[step], delta_0_3[step])
            )

        feas_constraints.append(Or(delta_0_0[0], delta_0_1[0], delta_0_2[0], delta_0_3[0]))

        return simplify(And(feas_constraints))

def example_obstacle():
    return Obstacle(
        3,
        [
            from_interval(np.array([0.1, 0.4]), np.array([0.7, 0.5])),
        ]
    )
