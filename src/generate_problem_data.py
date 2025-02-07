import pandas as pd
from tqdm import tqdm

from set.polyhedron import Polyhedron
from system.solvable import Solvable


def generate_problem_data(problem: Solvable, domain: Polyhedron, output_size: int, problem_data_output_file: str,
                          points: int):
    """
    Generates problem data by sampling points from the provided root polyhedron and solving for output.
    Saves results to a CSV file.

    Args:
        problem (Solvable): the optimization problem.
        domain (Polyhedron): the domain polyhedron.
        output_size (int): the predicted boolean list size.
        problem_data_output_file (str): the output file name.
        points (int): the number of points to sample.
    """
    columns = [f'theta_{var}' for var in range(domain.dim)] + [f'delta_{var}' for var in range(output_size)]
    df = pd.DataFrame(columns=columns)

    progress_bar = tqdm(desc='Generating Training Data', total=points, unit='point')

    i = 0
    while i < points:
        progress_bar.n = i + 1
        progress_bar.refresh()
        point = domain.random_point()

        mip_sol = problem.solve(point)
        sol = problem.get_output(mip_sol)

        if sol is None:
            continue

        i += 1
        row = [el for el in point] + (sol if mip_sol.Status == 2 else [-1] * output_size)

        row_df = pd.DataFrame([row], columns=columns)
        df = pd.concat([row_df, df], ignore_index=True) if not df.empty else row_df

    progress_bar.close()
    df.to_csv(problem_data_output_file, index=False)
    print(f"Problem data saved to {problem_data_output_file}")
