# -*- coding: utf-8 -*-

from data_loaders.get_a_game_data import GameData
from data_loaders.get_a_game_ball_data import BallData
from plotters.pitch_scatter import PitchScatter
from data_loaders.sqls import Sqls
from data_loaders.mine_sql_connection import MySqlConnection

import matplotlib.pyplot as plt
import numpy as np

import pickle
import pprint
import math
import time

TEAM = 0
SEC = 1
SPEED = 9
NAME = 11
X = 5
Y = 6


class PitchValue:
    X_DOWN = 0
    X_UP = 110
    Y_DOWN = 68
    Y_UP = 0
    X_RANGE = np.linspace(X_DOWN, X_UP, 50)
    Y_RANGE = np.linspace(Y_DOWN, Y_UP, 50)

    def __init__(self):
        self.ball_pos_data = []
        self.def_players_data_at_sec = []
        self.off_players_data_at_sec = []
        self.results = np.zeros((50, 50))
        with open('pickles/pitch_value_data2.pkl', 'rb+') as f:
            self.data = pickle.load(f)

        self.get_ball_pos()
        for each in self.data:
            try:
                print each[0][1]
            except:
                print "Problem"
        self.ball_pos_dict = {x[0]: (x[4], x[5]) for x in
                              filter(lambda x: x[0] in [x[0][1] for x in self.data], self.ball_pos_data)}

    @staticmethod
    def l2(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    @staticmethod
    def get_xy(sec):
        return sec[X], sec[Y]

    def sin(self, p1, p2):
        return (p2[1] - p1[1]) / self.l2(p1, p2)

    def cos(self, p1, p2):
        return (p2[0] - p1[0]) / self.l2(p1, p2)

    def get_ball_pos(self):
        conn = MySqlConnection()
        conn.execute_query(Sqls.GET_BALL_POS_DATA)
        for each in conn.get_cursor():
            self.ball_pos_data.append(each)

    def calculate_influence(self, sec1, sec2, ref_p=None, prnt=False, factor=1):
        if not ref_p:
            ref_p = self.get_xy(sec1)

        covit = self.calculate_covit(sec1, sec2)
        exp_part1 = ref_p - self.calculate_muit(sec1, sec2)
        exp_part2 = np.matmul(np.linalg.inv(self.calculate_covit(sec1, sec2)),
                              (ref_p - self.calculate_muit(sec1, sec2)))
        exp_part = math.exp(-0.5 * (np.matmul(exp_part1.transpose(), exp_part2)))
        inf_part1 = 1.0 / (((2 * math.pi) ** 2 * np.linalg.det(covit)) ** 0.5)

        if factor < 1:
            return min(((inf_part1 * exp_part) ** factor) / (3 / factor), 0.050)
        return inf_part1 * exp_part

    def calculate_covit(self, sec1, sec2):
        rotit = self.calculate_rotit(self.get_xy(sec1), self.get_xy(sec2))
        sit = self.calculate_sit(sec1)
        return reduce(np.matmul, [rotit, sit, sit, np.linalg.inv(rotit)])

    def calculate_rotit(self, p1, p2):
        return np.array([[self.cos(p1, p2), -self.sin(p1, p2)], [self.sin(p1, p2), self.cos(p1, p2)]])

    def calculate_sit(self, sec):
        rit_score = self.rit((sec[X], sec[Y]), self.ball_pos_dict[sec[SEC]])
        scaling = rit_score * (sec[SPEED] ** 2 / 169)

        return np.array([[(rit_score + scaling) / 2, 0], [0, (rit_score - scaling) / 2]])

    def rit(self, pp, bp):
        return min((self.l2(pp, bp) ** 3) / 1000.0 + 4, 10)

    def calculate_muit(self, sec1, sec2):
        return np.array(
            [self.get_xy(sec1)[0] + (sec2[X] - sec1[X]) / 2, self.get_xy(sec1)[1] + (sec2[Y] - sec1[Y]) / 2])

    def get_defender_positions(self, sec):
        def_players = [x[NAME] for x in filter(lambda x: x[0] == 101, self.data[0])]
        self.def_players_data_at_sec = [[filter(lambda x: x[NAME] == z, y)[0] for y in self.data[sec:sec+2]] for z in def_players]

    def get_offense_positions(self, sec):
        print "#" * 50
        print "#" * 50
        for each in self.data[0]:
            print each
        print "#" * 50
        print "#" * 50
        off_players = [x[NAME] for x in filter(lambda x: x[0] == 3, self.data[0])]
        self.off_players_data_at_sec = [[filter(lambda x: x[NAME] == z, y)[0] for y in self.data[sec:sec+2]] for z in off_players]

    def initialize(self):
        for i, x in enumerate(self.X_RANGE):
            for j, y in enumerate(self.Y_RANGE):
                ball_x = self.ball_pos_dict[self.data[0][0][SEC]][0]
                self.results[i][j] = (x - ball_x) / 3000

    def add_defensive_players(self):
        for player in self.def_players_data_at_sec:
            sec1 = player[0]
            sec2 = player[1]
            pprint.pprint('%s POSITION %.3f-%.3f' % (sec1[NAME], self.get_xy(sec1)[0], self.get_xy(sec1)[1]))

            for i, x in enumerate(self.X_RANGE):
                for j, y in enumerate(self.Y_RANGE):
                    self.results[i][j] += self.calculate_influence(sec1, sec2, ref_p=[x, y], factor=0.3)

    def add_behind_of_players(self):
        for player in self.def_players_data_at_sec:
            sec1 = list(player[0])
            sec2 = list(player[1])
            sec1[X] += (108 - player[0][X]) / 2
            sec1[Y] += (34 - player[0][Y]) / 2
            sec2[X] += (108 - player[1][X]) / 2
            sec2[Y] += (34 - player[1][Y]) / 2
            pprint.pprint('%s POSITION %.3f-%.3f' % (sec1[NAME], self.get_xy(sec1)[0], self.get_xy(sec1)[1]))

            for i, x in enumerate(self.X_RANGE):
                for j, y in enumerate(self.Y_RANGE):
                    self.results[i][j] += self.calculate_influence(sec1, sec2, ref_p=[x, y], factor=0.5) / 11

    def add_goal(self):
        sec1 = [0] * 11
        sec1[Y] = 34
        sec1[X] = 110
        sec1[SPEED] = 0.1
        sec1[SEC] = self.data[4][0][SEC]
        sec2 = [0] * 11
        sec2[Y] = 34
        sec2[X] = 109.9
        sec2[SPEED] = 0
        sec2[SEC] = self.data[5][0][SEC]
        print "Add goal Position %.3f-%.3f" % (sec1[X], sec1[Y])

        for i, x in enumerate(self.X_RANGE):
            for j, y in enumerate(self.Y_RANGE):
                self.results[i][j] += self.calculate_influence(sec1, sec2, ref_p=[x, y], factor=0.2)

    def add_ball(self, sec):
        sec1 = [0] * 11
        sec1[X] = self.ball_pos_dict[self.data[sec][0][SEC]][0]
        sec1[Y] = self.ball_pos_dict[self.data[sec][0][SEC]][1]
        sec1[SPEED] = 0.1
        sec1[SEC] = self.data[sec][0][SEC]
        sec2 = [0] * 11
        sec2[X] = self.ball_pos_dict[self.data[sec+1][0][SEC]][0]
        sec2[Y] = self.ball_pos_dict[self.data[sec+1][0][SEC]][1]
        sec2[SPEED] = 0
        sec2[SEC] = self.data[sec+1][0][SEC]
        print "Add ball Position %.3f-%.3f" % (sec1[X], sec1[Y])

        for i, x in enumerate(self.X_RANGE):
            for j, y in enumerate(self.Y_RANGE):
                self.results[i][j] += self.calculate_influence(sec1, sec2, ref_p=[x, y], factor=0.3)

    def normalize(self):
        maxx = np.max(self.results)
        minn = np.min(self.results)

        for i, x in enumerate(self.X_RANGE):
            for j, y in enumerate(self.Y_RANGE):
                self.results[i][j] = (self.results[i][j] - minn) / (maxx - minn)

    def show(self, prep_results=None, sec=None):
        if prep_results is not None:
            results = np.array(prep_results).transpose()
            self.get_defender_positions(sec)
            self.get_offense_positions(sec)
        else:
            results = np.array(self.results).transpose()
        fig, axs = plt.subplots(1, 1)
        im = axs.imshow(results, extent=[0, 110, 0, 68])
        plt.colorbar(im)
        self.scat_players_and_ball(sec)
        # plt.savefig('defensive_influence.pdf')
        plt.show()

    def scat_players_and_ball(self, sec):
        for player in self.def_players_data_at_sec:
            x, y = self.get_xy(player[0])
            plt.scatter(x, y, c='red', s=100)
        for player in self.off_players_data_at_sec:
            x, y = self.get_xy(player[0])
            plt.scatter(x, y, c='green', s=100)

        plt.scatter(self.ball_pos_dict[self.data[sec][0][SEC]][0],
                    self.ball_pos_dict[self.data[sec][0][SEC]][1],
                    c='white', s=100)

    def show_all(self, showed_sec):
        self.get_defender_positions(showed_sec)
        self.get_offense_positions(showed_sec)
        self.initialize()
        self.add_defensive_players()
        # self.add_behind_of_players()
        self.add_goal()
        self.add_ball(showed_sec)
        self.normalize()
        return self.results
        # self.show()


# pv = PitchValue()
# values = []
# for i in range(1):
#     print "#" * 50
#     print "#" * 50
#     pprint.pprint("SECOND %d" % (i+1))
#     print "#" * 50
#     print "#" * 50
#     values.append(pv.show_all(i))
#
# for c, each in enumerate(values):
#     pv.show(each, c)
#     time.sleep(1)









# listt = [
#     [[70, 33], [71, 34], 10],
#     [[75, 30], [76, 31], 4],
#     [[80, 25], [81, 26], 3],
#     [[84, 20], [83, 21], 2],
#     [[84, 15], [85, 16], 1],
#     [[75, 11], [74, 10], 5],
#     [[70, 7], [71, 6], 40],
#     [[60, 7], [61, 6], 3],
#     [[55, 7], [54, 8], 2],
#     [[50, 15], [49, 16], 1],
#     [[45, 20], [46, 21], 5],
#     [[48, 24], [47, 23], 22],
#     [[55, 30], [54, 31], 3],
#     [[60, 33], [61, 32], 2]
# ]
# pv = PitchValue()
#
# ball = [0] * 11
# ball[X] = 68.07
# ball[Y] = 18.90
#
# for each in listt:
#     sec1 = [0] * 11
#     sec1[X] = each[0][0]
#     sec1[Y] = each[0][1]
#     sec1[SPEED] = each[2]
#     sec1[SEC] = 0
#     sec2 = [0] * 11
#     sec2[X] = each[1][0]
#     sec2[Y] = each[1][1]
#     sec2[SPEED] = 1.60
#     sec2[SEC] = 1
#     pv.ball_pos_dict[0] = ball
#
#     print "Add scenario Position %.3f-%.3f" % (sec1[X], sec1[Y])
#     for i, x in enumerate(pv.X_RANGE):
#         for j, y in enumerate(pv.Y_RANGE):
#             if abs(x - sec1[X]) + abs(y - sec1[Y]) < 3:
#                 print "%.2f: x, %.2f: y score: %.6f" % (x, y, pv.calculate_influence(sec1, sec2, ref_p=[x, y]))
#             pv.results[i][j] += pv.calculate_influence(sec1, sec2, ref_p=[x, y])
#
# pv.results = np.array(pv.results).transpose()
# fig, axs = plt.subplots(1, 1)
# im = axs.imshow(pv.results, extent=[0, 110, 0, 68])
# plt.colorbar(im)
# for each in listt:
#     plt.scatter(each[0][0], each[0][1], c='green', s=100)
#     plt.scatter(each[1][0], each[1][1], c='red', s=100)
#     plt.scatter(ball[X], ball[Y], c='white', s=100)
# plt.show()