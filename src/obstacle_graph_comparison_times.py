import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

COMPARISON_PLOT_OUTPUT_FILE = 'resources/verified_obstacle/obstacle_comparison_time.pdf'

TIMINGS_DATA_FILE = 'resources/verified_obstacle/timings_data.csv'

def ax_1():
    data = pd.read_csv(TIMINGS_DATA_FILE)

    # Create a figure and axis objects for each subplot
    fig = plt.figure(figsize=(4, 4))
    ax = fig.gca()

    t_miqp = data['t_miqp']
    t_miqp_cut = t_miqp.map(lambda t : 0.1 if t >= 0.1 else t)
    t_qp = data['t_qp']
    t_qp_cut = t_qp.map(lambda t : 0.1 if t >= 0.1 else t)

    line_x = [0, max(t_miqp_cut)]
    line_y = [0, max(t_miqp_cut)]

    ax.plot(line_x, line_y, linestyle='--', alpha=0.2)

    ax.scatter(t_qp, t_miqp, marker='x', color='k')
    ax.set_xlim(-0.005, 0.1)
    ax.set_ylim(-0.005, 0.1)

    plt.ylabel('Solving Time $(1)$ (s)')
    plt.xlabel('Solving Time $(2)$ (s)')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

    fig.tight_layout()

    # Display the plot
    plt.savefig(COMPARISON_PLOT_OUTPUT_FILE)

if __name__ == '__main__':
    ax_1()
    print(f'Saved timings plot to {COMPARISON_PLOT_OUTPUT_FILE}')
