from pitch_value import PitchValue
from pass_probability import PassProbability

import matplotlib.pyplot as plt
import numpy as np
import random
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

    def __init__(self, game_id, home_id, away_id):
        self.pv = PitchValue(home_id, away_id, game_id)
        self.pp = PassProbability(home_id, away_id, game_id)
        self.UHVA = None
        self.frobenious = 0
        self.mean_x = 0
        self.mean_y = 0
        self.r = 0
        self.pp_result = 0
        self.pv_result = 0
        self.game_id = game_id
        self.home_id = home_id
        self.away_id = away_id

    @staticmethod
    def l2(p1, p2):
        return ((p1[X] - p2[X]) ** 2 + (p1[Y] - p2[Y]) ** 2) ** 0.5

    @staticmethod
    def l22(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def calculate_UHVA(self, sec=0, off_data=[], def_data=[], ref_p=[]):
        self.pv_result = self.pv.show_all(off_data, def_data, ref_p).transpose()
        print "PV, sec: %d" % sec
        self.pp_result = self.pp.show_all(off_data, def_data, ref_p).transpose()
        print "PP, sec: %d" % sec
        self.UHVA = np.add(self.pv_result, self.pp_result*0.8)
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

    def set_frob(self, frob_val):
        self.frobenious = frob_val

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

    def show(self, pre_result=None, sec=0):
        # self.pv.get_defender_positions(c)
        # self.pv.get_offense_positions(c)

        fig, axs = plt.subplots(1, 1, figsize=(14, 6))
        if pre_result is not None:
            self.UHVA = pre_result

        plt.axis('off')

        im = axs.imshow(self.UHVA, extent=[0, 110, 0, 68])
        plt.colorbar(im)
        img = plt.imread("resources/pitch.png")
        axs.imshow(img, extent=[0, 110, 0, 68], alpha=0.6)
        # plt.colorbar(im)

        for player in self.pv.def_players_data_at_sec:
            x, y = self.pv.get_xy(player[0])
            axs.scatter(x, y, c='orange', s=100)
        for player in self.pv.def_players_data_at_sec:
            x, y = self.pv.get_xy(player[1])
            axs.scatter(x, y, c='red', s=100)


        # self.pv.get_defender_positions(c+2)
        # self.pv.get_offense_positions(c+2)

        for player in self.pv.off_players_data_at_sec:
            x, y = self.pv.get_xy(player[0])
            axs.scatter(x, y, c='yellow', s=100)
        for player in self.pv.off_players_data_at_sec:
            x, y = self.pv.get_xy(player[1])
            axs.scatter(x, y, c='green', s=100)

        # axs.scatter(self.pv.get_llajic_pos(c-2)[0], self.pv.get_llajic_pos(c-2)[1], c='cyan', s=130)
        # axs.scatter(self.pv.get_llajic_pos(c)[0], self.pv.get_llajic_pos(c)[1], c='blue', s=130)
        # axs.scatter(self.pv.get_llajic_pos(c+2)[0], self.pv.get_llajic_pos(c+2)[1], c='black', s=130)

        axs.scatter(self.pv.ball_pos_dict[sec][0],
                    self.pv.ball_pos_dict[sec][1],
                    c='white', s=100)

        plt.show()
        # # plt.savefig("plots/uhva.eps", bbox_inches='tight')
        #

# uhva = UHVA()
# uhvas = []
# X_DOWN = 0
# X_UP = 110
# Y_DOWN = 68
# Y_UP = 0
# X_RANGE = np.linspace(X_DOWN, X_UP, 50)
# Y_RANGE = np.linspace(Y_DOWN, Y_UP, 50)
#
# for i in range(9):
#     try:
#         uhva.frob(i)
#         uhva.mean()
#         uhva.calculate_UHVA(i)
#
#         # llajic_dist = 150
#         # for j, x in enumerate(X_RANGE):
#         #     for k, y in enumerate(Y_RANGE):
#         #         if llajic_dist > UHVA.l22([x, y], uhva.pv.get_llajic_pos(i)):
#         #             llajic_dist = UHVA.l22([x, y], uhva.pv.get_llajic_pos(i))
#         #             gen_j = j
#         #             gen_k = k
#         #
#         # print "Llajic score: %.2f sec: %d" % (uhva.UHVA[gen_j][gen_k], i)
#         uhvas.append(uhva.UHVA)
#         uhva.show(c=i)
#     except:
#         import traceback
#         traceback.print_exc()
#     # break
#
# # for c, each in enumerate(uhvas):
# #     time.sleep(1)
