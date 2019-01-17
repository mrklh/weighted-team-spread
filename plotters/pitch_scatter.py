import numpy as np
import matplotlib.pyplot as plt
import pprint


class PitchScatter:
    def __init__(self, teams):
        self.teams = teams
        self.fig = None
        self.ax = None

    def set_fig_ax(self, fig, ax):
        self.fig = fig
        self.ax = ax

    def prepare_plot(self):
        img = plt.imread("resources/football_pitch.png")
        self.ax.imshow(img, extent=[0, 115, 0, 70])
        self.ax.set_xticks(np.arange(-2.5, 107.5, 5))
        self.ax.set_yticks(np.arange(-2.5, 70.5, 5))
        self.fig.tight_layout()

    def display_scatter(self, importance_list):
        self.prepare_plot()
        means = self.display_players_avg(importance_list)
        self.display_player_names(means)

    def display_players_avg(self, importance_list):
        means = []
        for team in self.teams:
            means.append(team.get_means())

        colors = ['#ff0000', '#00ff00']
        for i, mean in enumerate(means):
            s = [(im/2.0)**3 for im in importance_list[i]]
            plt.scatter(mean[:, 0], mean[:, 1], color=colors[i], s=s)

        return means

    def display_player_names(self, means):
        for i, team in enumerate(self.teams):
            for j, key in enumerate(team.get_player_names()):
                self.ax.annotate(key, (means[i][j,0] + 1, means[i][j,1] - 1))