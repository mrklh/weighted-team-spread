# from pitch_value import PitchValue
# from pass_probability import PassProbability

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import pprint
import time

TEAM = 0
SEC = 1
SPEED = 9
NAME = 11
X = 5
Y = 6
FROB_RATIO = 153.013360979


class UHVA:
    X_DOWN = 0
    X_UP = 110
    Y_DOWN = 68
    Y_UP = 0
    X_RANGE = np.linspace(X_DOWN, X_UP, 50)
    Y_RANGE = np.linspace(Y_DOWN, Y_UP, 50)

    def __init__(self):
        self.pv = PitchValue()
        self.pp = PassProbability()
        self.UHVA = None
        self.frobenious = 0
        self.mean_x = 0
        self.mean_y = 0
        self.r = 0
        self.pp_result = 0
        self.pv_result = 0

    @staticmethod
    def l2(p1, p2):
        return ((p1[X] - p2[X]) ** 2 + (p1[Y] - p2[Y]) ** 2) ** 0.5

    @staticmethod
    def l22(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def calculate_UHVA(self, sec=0):
        self.pv_result = self.pv.show_all(sec).transpose()
        self.pp_result = self.pp.show_all(sec).transpose()
        self.UHVA = np.add(self.pv_result, self.pp_result*1.5)
        self.normalize()
        self.apply_mean()
        self.normalize()
        # self.UHVA = self.UHVA.transpose()

    def normalize(self):
        maxx = np.max(self.UHVA)
        minn = np.min(self.UHVA)

        for i, x in enumerate(self.X_RANGE):
            for j, y in enumerate(self.Y_RANGE):
                self.UHVA[i][j] = (self.UHVA[i][j] - minn) / (maxx - minn)

    def frob(self, sec):
        self.pv.get_offense_positions(sec)
        for player1 in self.pv.off_players_data_at_sec:
            for player2 in self.pv.off_players_data_at_sec:
                self.frobenious += self.l2(player1[0], player2[0])

    def mean(self):
        x = 0
        y = 0
        for player in self.pv.off_players_data_at_sec:
            x += player[0][X]
            y += player[0][Y]

        self.mean_x = x/11
        self.mean_y = y/11
        self.r = 7000/FROB_RATIO

    def apply_mean(self):
        for i, x in enumerate(self.X_RANGE):
            for j, y in enumerate(self.Y_RANGE):
                diff = self.l22([self.mean_x, self.mean_y], [x, y])
                if diff > self.r:
                    self.UHVA[i][j] *= self.r/diff

    def show(self, pre_result=None, c=0):
        self.pv.get_defender_positions(c)
        self.pv.get_offense_positions(c)

        fig, axs = plt.subplots(1, 1, figsize=(14, 6))
        if pre_result is not None:
            self.UHVA = pre_result

        # im = axs[0].imshow(self.pp_result, extent=[0, 110, 0, 68])
        # plt.colorbar(im)

        # im = axs[1].imshow(self.pv_result, extent=[0, 110, 0, 68])
        # plt.colorbar(im)
        #
        im = axs.imshow(self.UHVA, extent=[0, 110, 0, 68])
        plt.colorbar(im)

        for player in self.pv.def_players_data_at_sec:
            x, y = self.pv.get_xy(player[0])
            axs.scatter(x, y, c='red', s=75)
            # axs[1].scatter(x, y, c='red', s=30)
            # axs[2].scatter(x, y, c='red', s=30)
        for player in self.pv.off_players_data_at_sec:
            x, y = self.pv.get_xy(player[0])
            axs.scatter(x, y, c='green', s=75)
            # axs[1].scatter(x, y, c='green', s=30)
            # axs[2].scatter(x, y, c='green', s=30)

        # axs[0].scatter(self.mean_x, self.mean_y, c='blue', s=150)
        # axs[1].scatter(self.mean_x, self.mean_y, c='blue', s=150)
        # axs[2].scatter(self.mean_x, self.mean_y, c='blue', s=150)

        axs.scatter(self.pv.ball_pos_dict[self.pv.data[c][0][SEC]][0],
                    self.pv.ball_pos_dict[self.pv.data[c][0][SEC]][1],
                    c='white', s=75)
        # axs[1].scatter(self.pv.ball_pos_dict[self.pv.data[c][0][SEC]][0],
        #             self.pv.ball_pos_dict[self.pv.data[c][0][SEC]][1],
        #             c='white', s=30)
        # axs[2].scatter(self.pv.ball_pos_dict[self.pv.data[c][0][SEC]][0],
        #             self.pv.ball_pos_dict[self.pv.data[c][0][SEC]][1],
        #             c='white', s=30)

        # plt.grid()
        # plt.show()
        plt.savefig("plots/final_sprint.pdf", bbox_inches='tight')


# uhva = UHVA()
# uhvas = []
# for i in range(9):
#     uhva.frob(i)
#     uhva.mean()
#     uhva.calculate_UHVA(i)
#     uhvas.append(uhva.UHVA)
#     uhva.show(c=i)

# for c, each in enumerate(uhvas):
#     time.sleep(1)


a = [[25, 36],
     [48, 65],
     [51, 50],
     [50, 35],
     [46, 10],
     [55, 30],
     [57, 8],
     [68, 39],
     [65, 33],
     [65, 15],
     [10, 32]]

b = [[48, 32],
     [55, 36],
     [59, 20],
     [65, 42],
     [68, 32],
     [72, 65],
     [75, 40],
     [84, 42],
     [79, 33],
     [87, 5],
     [20, 34]]


fig, ax = plt.subplots()
img = plt.imread("C:/Users/kulah/Desktop/pitchbw.png")
ax.imshow(img, extent=[0, 110, 0, 68])
ax.set_xticks(np.arange(-2.5, 110, 10))
ax.set_yticks(np.arange(-2.5, 67.5, 5))
fig.tight_layout()
plt.scatter([x[0] for x in b], [y[1] for y in b], color='b', s=100)
rect = patches.Rectangle((46, 66), 42, -63, linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)

plt.text(46.5, 10, "60 meters", {'color': 'r'})
plt.text(56, 63, "39 meters", {'color': 'r'})
# ax.annotate("60 meters", (36, 20))
# ax.annotate("40 meters", (50, 63))
# plt.show()
plt.savefig("bjkkon_far.pdf", bbox_inches="tight")