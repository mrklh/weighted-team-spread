from pitch_value_with_range import PitchValue
from pass_probability_with_range import PassProbability

import matplotlib.pyplot as plt
import numpy as np
import random
import math
import pprint
import time
import pickle
import pandas
import os

TEAM = 0
SEC = 1
PURE_SEC = 4
SPEED = 9
NAME = 11
X = 5
Y = 6
JERSEY = 7

FROB_RATIO = 153.013360979
SIZE_X = 100
SIZE_Y = 60

S_JERSEY = 3
S_END_SEC = 13
S_BEGIN_X = 14
S_BEGIN_Y = 15
S_END_X = 16
S_END_Y = 17


class UHVA:
    X_DOWN = 0
    X_UP = 110
    Y_DOWN = 68
    Y_UP = 0
    X_RANGE = np.linspace(X_DOWN, X_UP, SIZE_X)
    Y_RANGE = np.linspace(Y_DOWN, Y_UP, SIZE_Y)

    def __init__(self, game_id, home_id, away_id):
        self.pv = PitchValue(home_id, away_id, game_id)
        self.pp = PassProbability(home_id, away_id, game_id)
        self.UHVA = None
        self.mean_x = 0
        self.mean_y = 0
        self.r = 0
        self.pp_result = 0
        self.pv_result = 0
        self.game_id = game_id
        self.home_id = home_id
        self.away_id = away_id

        self.wts_list = UHVA.create_optimum_wts_list()

    @staticmethod
    def create_optimum_wts_list():
        unit_wts = 0.45484793365641374
        df = pandas.read_csv("team_spreads.txt", delimiter="\t")

        averages = [((x + y) * unit_wts) / 2 for x, y in zip(df['SOT'], df['CRS'])]
        return {x: y for x, y in zip(df['Team'], averages)}

    @staticmethod
    def l2(p1, p2):
        return ((p1[X] - p2[X]) ** 2 + (p1[Y] - p2[Y]) ** 2) ** 0.5

    @staticmethod
    def l22(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def calculate_UHVA(self, is_away, off_data, def_data, attack_team_id, sprint):
        def time_diff(sec1, sec2):
            if sec1 > sec2:
                return sec2 + 60 - sec1
            elif sec1 == sec2:  # to avoid -1 power in calculate_value function (**sec_diff - 1)
                return 1
            return sec2 - sec1

        self.pv_result = self.pv.show_all(off_data, def_data, is_away)
        self.pp_result = self.pp.show_all(off_data, def_data)
        self.UHVA = np.add(self.pv_result, self.pp_result)
        self.mean()
        self.apply_frob(attack_team_id)
        self.normalize()

        sprinter = filter(lambda x: x[0][JERSEY] == sprint[S_JERSEY], self.pv.off_players_data_at_sec)[0]

        value = self.calculate_value([sprint[S_BEGIN_X], sprint[S_BEGIN_Y]],
                                     [sprint[S_END_X], sprint[S_END_Y]],
                                     [sprinter[1][X], sprinter[1][Y]],
                                     time_diff(sprinter[1][PURE_SEC], sprint[S_END_SEC]))

        self.UHVA = self.UHVA.transpose()

        return value

    def normalize(self):
        maxx = np.max(self.UHVA)
        minn = np.min(self.UHVA)

        for i, x in enumerate(self.X_RANGE):
            for j, y in enumerate(self.Y_RANGE):
                self.UHVA[i][j] = (self.UHVA[i][j] - minn) / (maxx - minn)

    def mean(self):
        x = 0
        y = 0
        for player in self.pv.off_players_data_at_sec:
            x += player[0][X]
            y += player[0][Y]

        self.mean_x = x / 11
        self.mean_y = y / 11

    def apply_frob(self, attack_team_id):
        for i, x in enumerate(self.X_RANGE):
            for j, y in enumerate(self.Y_RANGE):
                diff = self.l22([self.mean_x, self.mean_y], [x, y])
                self.UHVA[i][j] *= min(1, self.wts_list[attack_team_id] / diff)

    def calculate_value(self, first_pos, second_pos, one_after_pos, sec_diff):
        # find closest points
        first_min_diff = 10
        second_min_diff = 10
        one_after_min_diff = 10
        first_i = 0
        first_j = 0
        one_after_i = 0
        one_after_j = 0
        second_i = 0
        second_j = 0
        for i, x in enumerate(self.X_RANGE):
            for j, y in enumerate(self.Y_RANGE):
                if self.l22(first_pos, [x, y]) < first_min_diff:
                    first_i = i
                    first_j = j
                    first_min_diff = self.l22(first_pos, [x, y])
                if self.l22(one_after_pos, [x, y]) < one_after_min_diff:
                    one_after_i = i
                    one_after_j = j
                    one_after_min_diff = self.l22(one_after_pos, [x, y])
                if self.l22(second_pos, [x, y]) < second_min_diff:
                    second_i = i
                    second_j = j
                    second_min_diff = self.l22(second_pos, [x, y])

        sprint_end_value = (self.UHVA[second_i][second_j] - self.UHVA[first_i][first_j]) / 2 + 0.5
        one_after_value = (self.UHVA[one_after_i][one_after_j] - self.UHVA[first_i][first_j]) / 2 + 0.5
        return one_after_value * (1 - 0.9 ** (sec_diff - 1)) + sprint_end_value * (0.9 ** (sec_diff - 1))

    def show(self, spc=0, value=0.0, sprint=None, game=0):
        fig, axs = plt.subplots(1, 1, figsize=(14, 6))
        plt.plot(5, 5, 20, 20, color='red', linewidth=5)
        plt.axis('off')

        im = axs.imshow(self.UHVA, extent=[0, 110, 0, 68])
        plt.colorbar(im)
        img = plt.imread("resources/pitch.png")
        axs.imshow(img, extent=[0, 110, 0, 68], alpha=0.6)

        for player in self.pv.def_players_data_at_sec:
            x1, y1 = self.pv.get_xy(player[0])
            x2, y2 = self.pv.get_xy(player[1])
            axs.scatter(x1, y1, c='orange', s=100)
            axs.scatter(x2, y2, c='red', s=100)

        for player in self.pv.off_players_data_at_sec:
            x1, y1 = self.pv.get_xy(player[0])
            x2, y2 = self.pv.get_xy(player[1])
            axs.scatter(x1, y1, c='yellow', s=100)
            axs.scatter(x2, y2, c='green', s=100)

        axs.scatter(sprint[S_BEGIN_X], sprint[S_BEGIN_Y], c='purple', s=150)
        axs.scatter(sprint[S_END_X], sprint[S_END_Y], c='cyan', s=150)

        ball_pos = PitchValue.to21_array(self.pv.ball_pos_dict[self.pv.off_players_data_at_sec[0][0][SEC]])
        axs.scatter(ball_pos[0][0], ball_pos[1][0], c='white', s=100)
        plt.title("Value: %.3f" % value)

        # plt.show()
        if not os.path.exists("plots/%d" % game):
            os.mkdir("plots/%d" % game)
        plt.savefig("plots/%d/uhva_%d_%.2f.png" % (game, spc, value), bbox_inches='tight')
        plt.close()

    def temp_show(self, def_players, off_players, uhva, value=0.0, sprint=None):
        fig, axs = plt.subplots(1, 1, figsize=(14, 6))
        plt.plot(5, 5, 20, 20, color='red', linewidth=5)
        plt.axis('off')

        im = axs.imshow(uhva, extent=[0, 110, 0, 68])
        plt.colorbar(im)
        img = plt.imread("resources/pitch.png")
        axs.imshow(img, extent=[0, 110, 0, 68], alpha=0.6)

        for player in def_players:
            x1, y1 = self.pv.get_xy(player[0])
            x2, y2 = self.pv.get_xy(player[1])
            axs.scatter(x1, y1, c='orange', s=100)
            axs.scatter(x2, y2, c='red', s=100)

        for player in off_players:
            x1, y1 = self.pv.get_xy(player[0])
            x2, y2 = self.pv.get_xy(player[1])
            axs.scatter(x1, y1, c='yellow', s=100)
            axs.scatter(x2, y2, c='green', s=100)

        axs.scatter(sprint[S_BEGIN_X], sprint[S_BEGIN_Y], c='purple', s=150)
        axs.scatter(sprint[S_END_X], sprint[S_END_Y], c='cyan', s=150)

        # ball_pos = PitchValue.to21_array(self.pv.ball_pos_dict[off_players[0][0][SEC]])
        # axs.scatter(ball_pos[0][0], ball_pos[1][0], c='white', s=100)
        plt.title("Value: %.3f" % value)

        plt.show()
        plt.close()

# uhva = UHVA(1, [], [])
