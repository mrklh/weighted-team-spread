import matplotlib.pyplot as plt
import numpy as np


class DistPlotter:
    def __init__(self, sec_data, team_ids, normal_dists, my_dists):
        self.sec_data = sec_data
        self.team_ids = team_ids
        self.normal_dists = normal_dists
        self.my_dists = my_dists
        self.fig = None
        self.ax = None
        self.home_sec_data = None
        self.away_sec_data = None

        self.initialize_variables()

    def initialize_variables(self):
        self.home_sec_data = filter(lambda x: x[0] == self.team_ids[0], self.sec_data)
        self.away_sec_data = filter(lambda x: x[0] == self.team_ids[1], self.sec_data)

    def set_fig_ax(self, fig, ax):
        self.fig = fig
        self.ax = ax

    def get_home_data(self):
        return [sec[5]+3 for sec in self.home_sec_data], \
               [sec[6] for sec in self.home_sec_data], \
               [sec[-1] for sec in self.home_sec_data]

    def get_away_data(self):
        return [sec[5]+3 for sec in self.away_sec_data], \
               [sec[6] for sec in self.away_sec_data], \
               [sec[-1] for sec in self.away_sec_data]

    def plot_pitch(self):
        fig, ax = plt.subplots()
        self.set_fig_ax(fig, ax)

        img = plt.imread("resources/football_pitch.png")
        self.ax.imshow(img, extent=[0, 115, 0, 72])
        self.ax.set_xticks(np.arange(-2.5, 107.5, 5))
        self.ax.set_yticks(np.arange(-2.5, 72.5, 5))
        self.fig.tight_layout()

    def put_players_on_pitch(self):
        x, y, names = self.get_home_data()
        self.ax.plot(x, y, 'ro')

        for c, name in enumerate(names):
            self.ax.text(x[c] + 1, y[c] - 1, name or 'Unknown', color='red')

        x, y, names = self.get_away_data()
        self.ax.plot(x, y, 'bo')

        for c, name in enumerate(names):
            self.ax.text(x[c] + 1, y[c] - 1, name or 'Unknown', color='blue')

    def plot_norm_dist(self):
        for i in range(2):
            if i == 0:
                data = self.home_sec_data
            else:
                data = self.away_sec_data

            if i == 0:
                min_x_ind = [x[5] for x in data].index(sorted([x[5] for x in data])[1])
                max_x_ind = [x[5] for x in data].index(sorted([x[5] for x in data])[-1])
            else:
                min_x_ind = [x[5] for x in data].index(sorted([x[5] for x in data])[-2])
                max_x_ind = [x[5] for x in data].index(sorted([x[5] for x in data])[0])

            color = 'r--' if i == 0 else 'b--'

            self.ax.plot([data[min_x_ind][5]+3, data[max_x_ind][5]+3],
                         [data[min_x_ind][6], data[min_x_ind][6]], color)

    def show_dist_stats(self):
        self.ax.text(5, 7, 'Normal distance between farthest players: %.2f meters' % self.normal_dists[0])
        self.ax.text(5, 10, 'My distance between farthest players      : %.2f mydist' % self.my_dists[0])
        self.ax.text(75, 7, 'Normal distance between farthest players: %.2f meters' % self.normal_dists[1])
        self.ax.text(75, 10, 'My distance between farthest players      : %.2f mydist' % self.my_dists[1])

    def show(self):
        plt.show()
