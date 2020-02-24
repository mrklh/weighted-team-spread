# -*- coding: utf-8 -*-

from data_loaders.get_a_game_data import GameData
from data_loaders.get_a_game_ball_data import BallData
from plotters.pitch_scatter import PitchScatter
from data_loaders.sqls import Sqls
from data_loaders.mine_sql_connection import MySqlConnection

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


BALL_SPEED = 15

print_rotit = False
print_sit = False
print_muit = False
print_covit = False
print_rit = False

class PassProbability:
    X_DOWN = 0
    X_UP = 110
    Y_DOWN = 68
    Y_UP = 0

    def __init__(self, home_id, away_id, game_id):
        self.ball_pos_data = []
        self.def_players_data_at_sec = []
        self.off_players_data_at_sec = []
        self.result = 0
        with open('pickles/pitch_value_data.pkl', 'rb+') as f:
            self.data = pickle.load(f)

        self.home_id = home_id
        self.away_id = away_id
        self.game_id = game_id

        self.get_ball_pos()
        self.ball_pos_dict = {x[0]: (x[4], x[5]) for x in self.ball_pos_data}

    @staticmethod
    def l2(p1, p2, printt=False):
        # if printt:
        #     print "(p1[0] - p2[0]) ** 2", (p1[0] - p2[0]) ** 2
        #     print "(p1[1] - p2[1]) ** 2", (p1[1] - p2[1]) ** 2
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
        conn.execute_query(Sqls.GET_BALL_POS_DATA % self.game_id)
        for each in conn.get_cursor():
            self.ball_pos_data.append(each)

    def calculate_influence(self, sec1, sec2, ref_p=None, prnt=False, factor=1.0, printt=False):
        if not ref_p:
            ref_p = self.get_xy(sec1)

        covit = self.calculate_covit(sec1, sec2, printt=printt)
        # print "#" * 50
        # print "#" * 50
        # print "COVITTTTTTTTTTTTTT"
        # pprint.pprint(covit)
        # print "#" * 50
        # print "#" * 50
        exp_part1 = ref_p - self.calculate_muit(sec1, sec2)
        exp_part2 = np.matmul(np.linalg.inv(covit),
                              (ref_p - self.calculate_muit(sec1, sec2)))
        exp_part = math.exp(-0.5 * (np.matmul(exp_part1.transpose(), exp_part2)))
        inf_part1 = 1.0 / ((((2 * math.pi) ** 2) * np.linalg.det(covit)) ** 0.5)
        # if printt:
        #     print "covit", covit
        #     print "np.linalg.det(covit)**0.5", np.linalg.det(covit)**0.5
        #     print "inf_part1", inf_part1

        if factor < 1:
            return min(((inf_part1 * exp_part) ** factor) / (3 / factor), 0.050)
        return inf_part1 * exp_part

    def calculate_covit(self, sec1, sec2, printt=False):
        global print_covit
        rotit = self.calculate_rotit(self.get_xy(sec1), self.get_xy(sec2), printt)
        sit = self.calculate_sit(sec1, printt)
        # if not print_covit:
        #     print "calculate_covit"
        #     print "#" * 50
        #     print "#" * 50
        #     pprint.pprint(rotit)
        #     print "#" * 50
        #     pprint.pprint(sit)
        #     print "#" * 50
        #     print "#" * 50
        #     print_covit = True
        # if printt:
        #     print "rotit", rotit
        #     print "sit", sit
        return reduce(np.matmul, [rotit, sit, sit, np.linalg.inv(rotit)])

    def calculate_rotit(self, p1, p2, printt=False):
        global print_rotit
        # if printt:
        #     print "self.cos(p1, p2)", self.cos(p1, p2)
        #     print "self.sin(p1, p2)", self.sin(p1, p2)
        if not print_rotit:
            # print "calculate_rotit"
            # print np.array([[self.cos(p1, p2), -self.sin(p1, p2)], [self.sin(p1, p2), self.cos(p1, p2)]])
            print_rotit = True
        return np.array([[self.cos(p1, p2), -self.sin(p1, p2)], [self.sin(p1, p2), self.cos(p1, p2)]])

    def calculate_sit(self, sec, printt=False):
        global print_sit
        rit_score = self.calculate_rit((sec[X], sec[Y]), self.ball_pos_dict[sec[SEC]], printt)
        # if printt:
        #     print "SPEED", sec[SPEED]
        #     print "X Y", sec[X], sec[Y]
        speed = sec[SPEED][0]**2 + sec[SPEED][1]**2
        scaling = rit_score * (speed / 169.0)
        # if not print_sit:
        #     print "calculate_sit"
        #     print "#" * 50
        #     print "/" * 50
        #     pprint.pprint(speed)
        #     print "/" * 50
        #     pprint.pprint(rit_score)
        #     print "/" * 50
        #     pprint.pprint(scaling)
        #     print "/" * 50
        #     pprint.pprint(np.array([[(rit_score + scaling) / 2, 0], [0, (rit_score - scaling) / 2]]))
        #     print "/" * 50
        #     print "#" * 50
        #     print_sit = True

        return np.array([[(rit_score + scaling) / 2, 0], [0, (rit_score - scaling) / 2]])

    def calculate_rit(self, pp, bp, printt=False):
        global print_rit

        return math.sin((self.l2(pp, bp) * (2 / 110) - 1) ** 2)*10 + 4

    def calculate_muit(self, sec1, sec2):
        global print_muit
        if not print_muit:
            # print "calculate_muit"
            # print np.array([sec1[X] + (sec1[SPEED][0] / 2), sec1[Y] + ((sec1[SPEED][1]) / 2)])
            print_muit = True
        return np.array([sec1[X] + (sec1[SPEED][0] / 2), sec1[Y] + ((sec1[SPEED][1]) / 2)])

    # def get_defender_positions(self, sec):
    #     def_players = [x[NAME] for x in filter(lambda x: x[0] == 101, self.data[0])]
    #     # def_players = ['Ferhat']
    #     self.def_players_data_at_sec = [[filter(lambda x: x[NAME] == z, y)[0] for y in self.data[sec:sec+2]] for z in def_players]
    #
    # def get_offense_positions(self, sec):
    #     off_players = [x[NAME] for x in filter(lambda x: x[0] == 3, self.data[0])]
    #     self.off_players_data_at_sec = [[filter(lambda x: x[NAME] == z, y)[0] for y in self.data[sec:sec+2]] for z in off_players]

    def initialize(self, ref_p):
        ball_x = self.ball_pos_dict[self.data[0][0][SEC]][0]
        self.result += (ref_p[0] - ball_x) / 4000

    def add_defensive_players(self, ref_p):
        for c, player in enumerate(self.def_players_data_at_sec):
            sec1 = player[0]
            sec2 = player[1]
            ball = [0] * 11
            ball[X] = self.ball_pos_dict[sec1[SEC]][0]
            ball[Y] = self.ball_pos_dict[sec1[SEC]][1]

            diff = self.l2(self.get_xy(sec1), self.get_xy(ball))
            exxp = (-math.sin((diff*(2.0/110) - 1)**7) - math.sin(diff*(2.0/110) - 1))/1.75

            sec1_speed_x = sec1[SPEED] / 4.0 * self.cos(self.get_xy(sec1), self.get_xy(sec2)) + 15 * exxp * self.cos(
                self.get_xy(ball),
                self.get_xy(sec1))
            sec1_speed_y = sec1[SPEED] / 4.0 * self.sin(self.get_xy(sec1), self.get_xy(sec2)) + 15 * exxp * self.sin(
                self.get_xy(ball),
                self.get_xy(sec1))

            sec2[X] += BALL_SPEED * self.cos(self.get_xy(ball), self.get_xy(sec1)) * exxp
            sec2[Y] += BALL_SPEED * self.sin(self.get_xy(ball), self.get_xy(sec1)) * exxp
            # sec1[SPEED] = (sec1_speed_x ** 2 + sec1_speed_y ** 2) ** 0.5
            sec1[SPEED] = [sec1_speed_x, sec1_speed_y]
            printt = True
            self.result = 1 - self.calculate_influence(sec1, sec2, ref_p=ref_p, factor=0.3, printt=printt)
            printt = False
            # if c>0:
            #     break

    # def normalize(self):
    #     maxx = np.max(self.results)
    #     minn = np.min(self.results)
    #
    #     for i, x in enumerate(self.X_RANGE):
    #         for j, y in enumerate(self.Y_RANGE):
    #             self.results[i][j] = (self.results[i][j] - minn) / (maxx - minn)

    def show(self, prep_results=None, sec=None):
        if prep_results is not None:
            results = np.array(prep_results).transpose()
            self.get_defender_positions(sec)
            self.get_offense_positions(sec)
        else:
            results = np.array(self.results).transpose()
        fig, axs = plt.subplots(1, 1)
        axs.set_xticks(np.linspace(0, 110, 23))
        axs.set_yticks(np.linspace(0, 68, 13))
        im = axs.imshow(results, extent=[0, 110, 0, 68])
        plt.colorbar(im)
        self.scat_players_and_ball(sec)
        # plt.savefig('defensive_influence.pdf')
        plt.grid()
        plt.show()

    def scat_players_and_ball(self, sec):
        for c, player in enumerate(self.def_players_data_at_sec):
            x, y = self.get_xy(player[0])
            plt.scatter(x, y, c='red', s=100)
            # if c > 0:
            #     break
        # for player in self.off_players_data_at_sec:
        #     x, y = self.get_xy(player[0])
        #     plt.scatter(x, y, c='green', s=100)

        plt.scatter(self.ball_pos_dict[self.data[sec][0][SEC]][0],
                    self.ball_pos_dict[self.data[sec][0][SEC]][1],
                    c='white', s=100)

    def show_all(self, off_data, def_data, ref_p):
        self.def_players_data_at_sec = def_data
        self.off_players_data_at_sec = off_data
        # self.get_defender_positions(showed_sec)
        # self.get_offense_positions(showed_sec)
        # self.initialize()
        self.add_defensive_players(ref_p)
        # self.normalize()
        return self.result
        # self.show()

#
# pp = PassProbability()
# values = []
# for i in range(1):
#     print "#" * 50
#     print "#" * 50
#     pprint.pprint("SECOND %d" % (i+1))
#     print "#" * 50
#     print "#" * 50
#
#     values.append(pp.show_all(i))
#
# for c, each in enumerate(values):
#     pp.show(each, c)
#     time.sleep(1)


########################################################################
########                     SCENARIO                     ##############
########################################################################

# listt = [
#     [[60, 34], [60.01, 34.01]],
#     # [[75, 30], [76, 31]],
#     # [[80, 25], [81, 26]],
#     # [[84, 20], [83, 21]],
#     # [[84, 15], [85, 16]],
#     # [[75, 11], [74, 10]],
#     # [[70, 7], [71, 6]],
#     # [[60, 7], [61, 6]],
#     # [[55, 7], [54, 8]],
#     # [[50, 15], [49, 16]],
#     # [[45, 20], [46, 21]],
#     # [[48, 24], [47, 23]],
#     # [[55, 30], [54, 31]],
#     # [[60, 33], [61, 32]]
# ]
# pv = PassProbability()
#
# ball = [0] * 11
# ball[X] = 50.00
# ball[Y] = 34.00
#
# for each in listt:
#     # for i, x in enumerate(pv.X_RANGE):
#     #     for j, y in enumerate(pv.Y_RANGE):
#     #         pv.results[i][j] = 0
#     sec1 = [0] * 11
#     sec1[X] = each[0][0]
#     sec1[Y] = each[0][1]
#     sec1[SPEED] = 0.01
#     sec1[SEC] = 0
#     sec2 = [0] * 11
#     sec2[X] = each[1][0]
#     sec2[Y] = each[1][1]
#     sec2[SPEED] = 0.01
#     sec2[SEC] = 1
#     pv.ball_pos_dict[0] = [ball[X], ball[Y]]
#     # exxp = min(18.0, math.exp(1 / pv.l2(pv.get_xy(sec1), pv.get_xy(ball)) ** 0.5) / 2) - 0.4
#     diff = pv.l2(pv.get_xy(sec1), pv.get_xy(ball))
#     exxp = (-math.sin((diff*(2.0/110) - 1)**7) - math.sin(diff*(2.0/110) - 1))/1.75
#     print "DIFFFFFF", diff, "EXPPPPPP", exxp
#     print "!" * 50
#     print "!" * 50
#     pprint.pprint(sec1[SPEED])
#     pprint.pprint(pv.cos(pv.get_xy(sec1), pv.get_xy(sec2)))
#     pprint.pprint(sec1[SPEED] * pv.cos(pv.get_xy(sec1), pv.get_xy(sec2)))
#     print "!" * 50
#     print "!" * 50
#     pprint.pprint(sec1[SPEED])
#     pprint.pprint(pv.sin(pv.get_xy(sec1), pv.get_xy(sec2)))
#     pprint.pprint(sec1[SPEED] * pv.sin(pv.get_xy(sec1), pv.get_xy(sec2)))
#     print "!" * 50
#     print "!" * 50
#     sec1_speed_x = sec1[SPEED]/4.0 * pv.cos(pv.get_xy(sec1), pv.get_xy(sec2)) + 15 * exxp * pv.cos(pv.get_xy(ball),
#                                                                                         pv.get_xy(sec1))
#     sec1_speed_y = sec1[SPEED]/4.0 * pv.sin(pv.get_xy(sec1), pv.get_xy(sec2)) + 15 * exxp * pv.sin(pv.get_xy(ball),
#                                                                     pv.get_xy(sec1))
#
#     # exxp = 0.1
#     print "#" * 50
#     print "#" * 50
#     pprint.pprint(exxp)
#     pprint.pprint(sec1_speed_x)
#     pprint.pprint(sec1_speed_y)
#     print "#" * 50
#     print "#" * 50
#     sec2[X] += 18 * pv.cos(pv.get_xy(ball), pv.get_xy(sec1)) * exxp
#     sec2[Y] += 18 * pv.sin(pv.get_xy(ball), pv.get_xy(sec1)) * exxp
#     # sec1[SPEED] = (sec1_speed_x ** 2 + sec1_speed_y ** 2) ** 0.5
#     sec1[SPEED] = [sec1_speed_x, sec1_speed_y]
#     # sec1[SPEED] = 8
#     print "#" * 50
#     print "#" * 50
#     pprint.pprint(sec1[SPEED])
#     print "#" * 50
#     print "#" * 50
#
#     print "\n\nAdd scenario Position %.3f-%.3f" % (sec1[X], sec1[Y])
#     printed = False
#     for i, x in enumerate(pv.X_RANGE):
#         for j, y in enumerate(pv.Y_RANGE):
#         #     if abs(x - sec1[X]) + abs(y - sec1[Y]) < 2 and not printed:
#         #         print "%.2f: x, %.2f: y speed: %.2f score: %.6f" % (x, y, sec1[SPEED], pv.calculate_influence(sec1, sec2, ref_p=[x, y], printt=True))
#         #         printed = True
#             pv.results[i][j] += pv.calculate_influence(sec1, sec2, ref_p=[x, y])
#
# pv.results = np.array(pv.results).transpose()
# pv.normalize()
# pv.results = pv.results[10:35, 18:37]
#
# font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 14}
#
# matplotlib.rc('font', **font)
#
# fig, axs = plt.subplots(1, 1)
# im = axs.imshow(pv.results, extent=[40, 80, 22.2, 53.6])
# # im = axs.imshow(pv.results, extent=[0, 110, 0, 68])
# # plt.colorbar(im)
# plt.xlabel("X distance in meters")
# plt.ylabel("Y distance in meters")
# for each in listt:
#     plt.scatter(each[0][0], each[0][1], c='red', s=100)
#     plt.arrow(ball[X], ball[Y], 18, 0, edgecolor='white', facecolor='white', shape='full',
#               length_includes_head=True, head_width=2, head_length=2)
#     # plt.arrow(each[0][0], each[0][1], each[1][0] - each[0][0], each[1][1] - each[0][1], edgecolor='red', facecolor='red', shape='full',
#     #           length_includes_head=True, head_width=2, head_length=2)
#     # plt.scatter(each[1][0], each[1][1], c='red', s=100)
#     plt.scatter(ball[X], ball[Y], c='white', s=100)
#
# divider = make_axes_locatable(axs)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)
# # plt.show()
# plt.savefig("plots/standing_interception.eps", bbox_inches='tight')
