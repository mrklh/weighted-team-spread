import numpy as np
import matplotlib.pyplot as plt
import pprint
from commons import Commons

class PitchPlotter:
    def __init__(self, sec_data_close, sec_data_far, close_scores, far_scores, diff_matrix):
        self.sec_data_close = sec_data_close
        self.sec_data_far = sec_data_far
        self.close_scores = close_scores
        self.far_scores = far_scores
        self.diff_matrix = diff_matrix
        self.fig = None
        self.axes = None

    def prepare_plot(self):
        self.fig, self.axes = plt.subplots(1, 1)
        # img = plt.imread("resources/football_pitch.png")
        # self.axes[0].imshow(img, extent=[0, 115, 0, 70])
        # self.axes.imshow(img, extent=[0, 115, 0, 70])
        # self.axes[0].get_xaxis().set_visible(False)
        # self.axes[0].get_yaxis().set_visible(False)
        # self.axes.get_xaxis().set_visible(False)
        # self.axes.get_yaxis().set_visible(False)
        # self.axes[0].set_xticks(np.arange(-2.5, 107.5, 5))
        # self.axes[0].set_yticks(np.arange(-2.5, 70.5, 5))
        # self.axes.set_xticks(np.arange(-2.5, 107.5, 5))
        # self.axes.set_yticks(np.arange(-2.5, 70.5, 5))
        # plt.imshow(img, extent=[0, 115, 0, 70])

    def display_scatter(self, weight, close, passs, names):
        names = ['(%d) %s' % (x, y) for x, y in zip(Commons.bjk_kon_num, names)]
        self.prepare_plot()
        self.plot_matrix(weight, close, passs, names)
        # self.display_players()
        # self.plot_diff(names)
        plt.savefig('weight.pdf', bbox_inches='tight')
        # plt.show()

    def plot_matrix(self, weight, close, passs, names):
        im = self.axes.imshow(weight, cmap='Greens')
        plt.colorbar(im)
        # self.axes.set_title("Cohesion Matrix", size=15)
        self.axes.set_xticks(np.arange(0, 11, 1))
        self.axes.set_yticks(np.arange(0, 11, 1))
        self.axes.set_xticklabels(names)
        self.axes.set_yticklabels(names)
        for tick in self.axes.get_xticklabels():
            tick.set_rotation(90)

        # im = self.axes.imshow(close, cmap='Greens')
        # plt.colorbar(im)
        # self.axes.set_title("Close Matrix", size=15)
        # self.axes.set_xticks(np.arange(0, 11, 1))
        # self.axes.set_yticks(np.arange(0, 11, 1))
        # self.axes.set_xticklabels(names)
        # self.axes.set_yticklabels(names)
        # for tick in self.axes.get_xticklabels():
        #    tick.set_rotation(90)
        #
        # im = self.axes[2].imshow(passs, cmap='Greens')
        # # plt.colorbar(im)
        # self.axes[2].set_title("Pass Matrix", size=15)
        # self.axes[2].set_xticks(np.arange(0, 11, 1))
        # self.axes[2].set_yticks(np.arange(0, 11, 1))
        # self.axes[2].set_xticklabels(names)
        # self.axes[2].set_yticklabels(names)
        # for tick in self.axes[2].get_xticklabels():
        #     tick.set_rotation(90)

    def plot_diff(self, names):
        im = self.axes.imshow(self.diff_matrix, cmap='Greens')
        plt.colorbar(im)
        # self.axes[1].set_title("Diff Matrix", size=15)
        self.axes.set_xticks(np.arange(0, 11, 1))
        self.axes.set_yticks(np.arange(0, 11, 1))
        self.axes.set_xticklabels(names)
        self.axes.set_yticklabels(names)
        for tick in self.axes.get_xticklabels():
            tick.set_rotation(90)

    def display_players(self):
        dtype = [('x', 'f'), ('y', 'f'), ('name', 'S21')]
        # print '#' * 49
        # print '#' * 49
        # pprint.pprint('Closest sec: %d' % self.sec_data_close[0][1])
        # pprint.pprint(self.close_scores)
        # pprint.pprint('Farthest sec: %d' % self.sec_data_far[0][1])
        # pprint.pprint(self.far_scores)
        # print '#' * 49
        # print '#' * 49
        players_close = np.rec.array([[x[5], 68-x[6], x[-1]] for x in filter(lambda x: x[0] == 3, self.sec_data_close)], dtype=dtype)
        players_far = np.rec.array([[x[5], 68-x[6], x[-1]] for x in filter(lambda x: x[0] == 3, self.sec_data_far)], dtype=dtype)

        # print '#' * 49
        # print '#' * 49
        # pprint.pprint(players_close)
        # pprint.pprint(self.close_scores)
        # print '#' * 49
        # pprint.pprint(players_far)
        # pprint.pprint(self.far_scores)
        # print '#' * 49
        # print '#' * 49
        # self.axes[0].scatter(players_close['x'], players_close['y'], s=700)
        # for player in players_close:
        #     off_x = 0.7
        #     if Commons.bjk_kon_jersey[player[2]] > 9:
        #         off_x = 1.6
        #     self.axes[0].text(player[0] - off_x, player[1] - 1, Commons.bjk_kon_jersey[player[2]], fontsize=12)

        self.axes.scatter(players_far['x'], players_far['y'], s=700)
        for player in players_far:
            off_x = 0.7
            if Commons.bjk_kon_jersey[player[2]] > 9:
                off_x = 1.6
            self.axes.text(player[0] - off_x, player[1] - 1, Commons.bjk_kon_jersey[player[2]], fontsize=12)
