import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


class DistTimePlotter:
    def __init__(self, secs_list_as_dict):
        self.secs_list_as_dict = secs_list_as_dict
        self.fig = None
        self.ax = None
        self.home_sec_data = None
        self.away_sec_data = None

        self.plot_time_dist_graph()
        self.show()

    def set_fig_ax(self, fig, ax):
        self.fig = fig
        self.ax = ax

    def plot_time_dist_graph(self):
        grid = gridspec.GridSpec(2, 1)
        fig = plt.figure()

        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = ax1.twinx()
        frob = [self.secs_list_as_dict[x][0]/2.0 for x in self.secs_list_as_dict] # frobenius
        mine = [self.secs_list_as_dict[x][1] for x in self.secs_list_as_dict] # mine
        # nrmx = [self.secs_list_as_dict[x][4] for x in self.secs_list_as_dict] # normx
        # nrmy = [self.secs_list_as_dict[x][5] for x in self.secs_list_as_dict] # normy

        frob_plot, = ax1.plot(np.arange(0, len(mine), 1), mine, 'r-', label="Frob.")
        mine_plot, = ax1.plot(np.arange(0, len(mine), 1), frob, 'g-', label="Mine")
        # nrmx_plot, = ax1.plot(np.arange(0, len(mine), 1), nrmx, 'b-', label="Norm X.")
        # nrmy_plot, = ax1.plot(np.arange(0, len(mine), 1), nrmy, 'k-', label="Norm Y.")

        keys = sorted(self.secs_list_as_dict.keys())
        for c, key in enumerate(keys):
            if keys[c - 1] != key - 1:
                ax1.axvline(x=c)

        plt.legend(handles=[frob_plot,
                            mine_plot,
                            # nrmx_plot,
                            # nrmy_plot
                           ]
        )
        ax1.set_title("Home Team")

        ax1 = fig.add_subplot(grid[1, 0])
        ax2 = ax1.twinx()
        my_away = [self.secs_list_as_dict[x][1] for x in self.secs_list_as_dict]
        norm_away = [self.secs_list_as_dict[x][3] for x in self.secs_list_as_dict]

        ax1.plot(np.arange(0, len(my_away), 1), my_away, 'r-')
        ax2.plot(np.arange(0, len(my_away), 1), norm_away, 'g-')
        ax1.set_title("Away Team")

    def show(self):
        plt.show()
