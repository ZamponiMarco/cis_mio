import math

import pandas as pd
from pandas import Series

TIMINGS_DATA_FILE = 'resources/verified_obstacle/timings_data.csv'
OUTPUT_TABLE_FILE = 'resources/verified_obstacle/latex_tables.tex'


def get_latex_table_feasibility(feas, feas_cegis, feas_dt, feas_rf):
    # Define the data entries
    data = {
        "MIP": [feas, 0],
        "CP": [feas_cegis, feas - feas_cegis],
        "DT": [feas_dt, feas - feas_dt],
        "RF": [feas_rf, feas - feas_rf]
    }

    # Generate LaTeX code
    latex_code = r"""
\begin{table}[t]
    \centering
    \begin{tabular}{ccc}
        \toprule
        \emph{Method} & \emph{\# Feasible} & \emph{\# Unfeasible} \\
        \midrule
"""

    for method, values in data.items():
        latex_code += f"        {method} & {values[0]} & {values[1]} \\\\\n"

    latex_code += r"""        \bottomrule
    \end{tabular}
    \vspace{1ex}
    \caption{Nonlinear system feasibility results for continuous problems, comparing the number of feasible instances when fixing binary values predicted by different methods.}
    \vspace{-0.5cm}
    \label{tab:obstacle-feasibility}
\end{table}
"""

    return latex_code


def get_latex_table_suboptimality(
        cegis_opt, cegis_1, cegis_5, cegis_5plus,
        dt_opt, dt_1, dt_5, dt_5plus,
        rf_opt, rf_1, rf_5, rf_5plus
):
    # Define the data entries
    data = {
        "CP": [cegis_opt, cegis_1, cegis_5, cegis_5plus],
        "DT": [dt_opt, dt_1, dt_5, dt_5plus],
        "RF": [rf_opt, rf_1, rf_5, rf_5plus]
    }

    # Generate LaTeX code
    latex_code = r"""
\begin{table}[t]
    \centering
    \begin{tabular}{ccccc}
        \toprule
        \emph{Method} & \emph{\# Opt} & \emph{\# <1\%} & \emph{\# <5\%} & \emph{\# $\geq$ 5\%} \\
        \midrule
"""

    for method, values in data.items():
        latex_code += f"        {method} & {values[0]} & {values[1]} & {values[2]} & {values[3]} \\\\\n"

    latex_code += r"""
        \bottomrule
    \end{tabular}
    \vspace{1ex}
    \caption{Nonlinear system suboptimality results for continuous problems, comparing the methods by the number of solutions within various suboptimality percentage ranges}
    \vspace{-0.5cm}
    \label{tab:obstacle-suboptimality}
\end{table}
"""

    return latex_code


def get_latex_code_timings(min_value, median_value, max_value):
    return f"""
\\begin{{table}}[t]
    \\centering
    \\begin{{tabular}}{{ccc}}
        \\toprule
        \\emph{{Minimum}} & \\emph{{Median}} & \\emph{{Maximum}} \\\\
        \\midrule
        {'{:.1f}'.format(min_value * 100)} \\% & {'{:.1f}'.format(median_value * 100)} \\% & {'{:.1f}'.format(max_value * 100)} \\% \\\\
        \\bottomrule
    \\end{{tabular}}
    \\vspace{{1ex}}
    \\caption{{Nonlinear system relative decrease in solving times of problem~\\eqref{{eq:mpMIP_deltafixed}} (fixed binary variables) relative to problem~\\eqref{{eq:mpMIP}}.}}
    \\vspace{{-0.5cm}}
    \\label{{tab:obstacle-times-table}}
\\end{{table}}
    """


if __name__ == '__main__':
    data = pd.read_csv(TIMINGS_DATA_FILE)

    my_cond = (data['s_miqp'] == 2)
    my_cond_2 = data['s_qp'] != 2

    cond = data['s_miqp'] == 2
    feas = data[cond]
    cond_fd = data['s_qp'] == 2
    feas_fd = data[cond_fd]
    cond_dt = data['s_qp_dt'] == 2
    ncond_dt = data['s_qp_dt'] != 2
    feas_dt = data[cond_dt]
    cond_rf = data['s_qp_rf'] == 2
    feas_rf = data[cond_rf]

    feasibility_latex = get_latex_table_feasibility(len(feas), len(feas[cond_fd]), len(feas[cond_dt]),
                                                    len(feas[cond_rf]))

    times = (feas['t_miqp'] - feas['t_qp'])/feas['t_miqp']
    times = times.describe()
    timings_latex = get_latex_code_timings(times['min'], times['50%'], times['max'])
    times = times.rename(
        {
            '': 'Test'
        }
    )

    subopt: Series = (feas['obj_qp'] - feas['obj_miqp'])
    subopt = subopt.clip(0, math.nan)
    subopt = subopt / feas['obj_miqp']

    subopt_dt: Series = (feas_dt['obj_qp_dt'] - feas_dt['obj_miqp'])
    subopt_dt = subopt_dt.clip(0, math.nan)
    subopt_dt = subopt_dt / feas_dt['obj_miqp']

    subopt_rf: Series = (feas_rf['obj_qp_rf'] - feas_rf['obj_miqp'])
    subopt_rf = subopt_rf.clip(0, math.nan)
    subopt_rf = subopt_rf / feas_rf['obj_miqp']

    suboptimality_latex = get_latex_table_suboptimality(
        len(subopt[subopt < 0.0001]),
        len(subopt[(subopt >= 0.0001) & (subopt < 0.01)]),
        len(subopt[(subopt >= 0.01) & (subopt < 0.05)]),
        len(subopt[(subopt >= 0.05)]),
        len(subopt_dt[subopt_dt < 0.0001]),
        len(subopt_dt[(subopt_dt >= 0.0001) & (subopt_dt < 0.01)]),
        len(subopt_dt[(subopt_dt >= 0.01) & (subopt_dt < 0.05)]),
        len(subopt_dt[(subopt_dt >= 0.05)]),
        len(subopt_rf[subopt_rf < 0.0001]),
        len(subopt_rf[(subopt_rf >= 0.0001) & (subopt_rf < 0.01)]),
        len(subopt_rf[(subopt_rf >= 0.01) & (subopt_rf < 0.05)]),
        len(subopt_rf[(subopt_rf >= 0.05)])
    )

    with open(OUTPUT_TABLE_FILE, "w") as f:
        f.write(feasibility_latex)
        f.write("\n\n")
        f.write(timings_latex)
        f.write("\n\n")
        f.write(suboptimality_latex)

    print(f'Saved result LaTeX table to {OUTPUT_TABLE_FILE}')
