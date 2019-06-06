import numpy as np
import matplotlib.pyplot as plt
import pprint


class PitchScatter:
    def __init__(self, teams=None, sec_data=None):
        self.teams = teams
        self.sec_data = sec_data
        self.fig = None
        self.ax = None

    def set_fig_ax(self, fig, ax):
        self.fig = fig
        self.ax = ax

    def prepare_plot(self):
        fig, ax = plt.subplots(1, 1)
        self.set_fig_ax(fig, ax)

        img = plt.imread("resources/football_pitch.png")
        self.ax.imshow(img, extent=[0, 115, 0, 70])
        self.ax.set_xticks(np.arange(-2.5, 107.5, 5))
        self.ax.set_yticks(np.arange(-2.5, 70.5, 5))
        self.fig.tight_layout()

    def display_scatter(self, importance_list=None):
        self.prepare_plot()
        if self.sec_data:
            self.display_player_positions()
        else:
            positions = self.display_players_avg(importance_list)
            self.display_player_names(positions)

        plt.show()


    def display_player_positions(self):
        colors = ['#ff0000', '#00ff00']
        home_id = self.teams[0].id

        for player in self.sec_data:
            plt.scatter(player[5], 68-player[6], color=colors[home_id == player[0]])
            self.ax.annotate(player[-1], (player[5] + 1, 68-player[6] - 1))


    def display_players_avg(self, importance_list):
        means = []
        for team in self.teams:
            means.append(team.get_means())

        colors = ['#ff0000', '#00ff00']
        for i, mean in enumerate(means):
            s = [(im/2.0)**3 for im in importance_list[i]]
            plt.scatter(mean[:, 0], mean[:, 1], color=colors[i], s=s)

        return means

    def display_player_names(self, positions):
        for i, team in enumerate(self.teams):
            for j, key in enumerate(team.get_player_names()):
                self.ax.annotate(key, (positions[i][j,0] + 1, positions[i][j,1] - 1))