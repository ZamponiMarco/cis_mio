import math
import random
import warnings

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from util.data_util import class_to_label
from examples.models.obstacle import Obstacle, example_obstacle
from tree.tree import BinaryTree

TIMINGS_DATA_FILE = 'resources/verified_obstacle/timings_data.csv'

POINTS = 1000

TREE_FILE_VERIFIED = 'resources/verified_obstacle/tree.pkl'
TREE_FILE_DT = 'resources/verified_obstacle/simple_tree.pkl'
TREE_FILE_RF = 'resources/verified_obstacle/forest_tree.pkl'

def get_problem_info_mip(point, model):
    solution_info = model.solve_function(point)
    return solution_info.Runtime, \
           solution_info.ObjVal if hasattr(solution_info, 'ObjVal') else math.nan, \
           solution_info.Status

def get_problem_info(point, prediction, model):
    solution_info = model.solve_fixed(point, prediction)
    return solution_info.Runtime, \
           solution_info.ObjVal if hasattr(solution_info, 'ObjVal') else math.nan, \
           solution_info.Status

if __name__ == '__main__':
    seed = random.randint(0, 2000000000)
    np.random.seed(seed)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    tree: BinaryTree = joblib.load(TREE_FILE_VERIFIED)
    clf_dt = joblib.load(TREE_FILE_DT)
    clf_rf = joblib.load(TREE_FILE_RF)

    obstacle: Obstacle = example_obstacle()

    columns = [f'theta_{var}'
               for var in range(tree.initial_polyhedron.dim)] + \
              ['t_miqp', 'obj_miqp', 's_miqp',
               't_qp', 'obj_qp', 's_qp',
               't_qp_dt', 'obj_qp_dt', 's_qp_dt',
               't_qp_rf', 'obj_qp_rf', 's_qp_rf',
               ]

    df = pd.DataFrame(columns=columns)

    progress_bar = tqdm(desc='Generating Comparison Data', total=POINTS, unit='point')

    i = 0
    while i < POINTS:
        progress_bar.n = i + 1
        progress_bar.refresh()
        point = tree.initial_polyhedron.random_point()

        mip_sol = obstacle.solve(point)

        if mip_sol.Status != 2:
            continue

        i += 1

        p_sol = obstacle.solve_fixed(point, class_to_label(tree.predict(point), tree.output_size))

        p_sol_dt = obstacle.solve_fixed(point, clf_dt.predict([point])[0])

        p_sol_rf = obstacle.solve_fixed(point, clf_rf.predict([point])[0])

        row = [el for el in point] + \
              [
                  mip_sol.Runtime, mip_sol.ObjVal if hasattr(mip_sol, 'ObjVal') else math.nan, mip_sol.Status,
                  p_sol.Runtime, p_sol.ObjVal if hasattr(p_sol, 'ObjVal') else math.nan, p_sol.Status,
                  p_sol_dt.Runtime, p_sol_dt.ObjVal if hasattr(p_sol_dt, 'ObjVal') else math.nan, p_sol_dt.Status,
                  p_sol_rf.Runtime, p_sol_rf.ObjVal if hasattr(p_sol_rf, 'ObjVal') else math.nan, p_sol_rf.Status,
              ]

        row_df = pd.DataFrame([row], columns=columns)

        if df.empty:
            df = row_df
        else:
            df = pd.concat([row_df, df], ignore_index=True)

    progress_bar.close()
    df.to_csv(TIMINGS_DATA_FILE)
    print(f'Saved timings data to {TIMINGS_DATA_FILE}')