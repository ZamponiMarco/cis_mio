import math
import random
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from util.data_util import class_to_label
from examples.models.obstacle_inertial_miosqp import solve_mpc_miosqp
from examples.models.obstacle_inertial_osqp import solve_mpc_osqp
from tree.tree import BinaryTree

TIMINGS_DATA_FILE = 'resources/verified_inertial_reduced/timings_data.csv'

POINTS = 1000

TREE_FILE_VERIFIED = 'resources/verified_inertial_reduced/tree.pkl'
TREE_FILE_DT = 'resources/verified_inertial_reduced/simple_tree.pkl'
TREE_FILE_RF = 'resources/verified_inertial_reduced/forest_tree.pkl'

if __name__ == '__main__':
    seed = random.randint(0, 2000000000)
    np.random.seed(seed)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    tree: BinaryTree = joblib.load(TREE_FILE_VERIFIED)
    clf_dt = joblib.load(TREE_FILE_DT)
    clf_rf = joblib.load(TREE_FILE_RF)

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

        try:
            mip_sol = solve_mpc_miosqp(point, 2)
        except TypeError as e:
            traceback.print_exc()
            continue

        if mip_sol.status != 'Solved':
            continue

        i += 1

        p_sol = solve_mpc_osqp(point, class_to_label(tree.predict(point), tree.output_size), 2)

        p_sol_dt = solve_mpc_osqp(point, clf_dt.predict([point])[0], 2)

        p_sol_rf = solve_mpc_osqp(point, clf_rf.predict([point])[0], 2)

        row = [el for el in point] + \
              [
                  mip_sol.run_time,
                  mip_sol.upper_glob if hasattr(mip_sol, 'upper_glob') else math.nan,
                  mip_sol.status,
                  p_sol.info.run_time,
                  p_sol.info.obj_val if hasattr(p_sol.info, 'obj_val') else math.nan,
                  p_sol.info.status,
                  p_sol_dt.info.run_time,
                  p_sol_dt.info.obj_val if hasattr(p_sol_dt.info, 'obj_val') else math.nan,
                  p_sol_dt.info.status,
                  p_sol_rf.info.run_time,
                  p_sol_rf.info.obj_val if hasattr(p_sol_rf.info, 'obj_val') else math.nan,
                  p_sol_rf.info.status,
              ]

        row_df = pd.DataFrame([row], columns=columns)

        if df.empty:
            df = row_df
        else:
            df = pd.concat([row_df, df], ignore_index=True)

    progress_bar.close()
    df.to_csv(TIMINGS_DATA_FILE)
    print(f'Saved timings data to {TIMINGS_DATA_FILE}')
