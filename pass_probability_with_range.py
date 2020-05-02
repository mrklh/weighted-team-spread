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

SIZE_X = 100
SIZE_Y = 60

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
    X_RANGE = np.linspace(X_DOWN, X_UP, SIZE_X)
    Y_RANGE = np.linspace(Y_DOWN, Y_UP, SIZE_Y)

    def __init__(self, home_id, away_id, game_id):
        self.ball_pos_data = []
        self.def_players_data_at_sec = []
        self.off_players_data_at_sec = []
        self.results = np.zeros((SIZE_X, SIZE_Y))
        with open('pickles/pitch_value_data.pkl', 'rb+') as f:
            self.data = pickle.load(f)

        self.home_id = home_id
        self.away_id = away_id
        self.game_id = game_id

        self.get_ball_pos()
        self.ball_pos_dict = {x[0]: (x[4], x[5]) for x in self.ball_pos_data}

    @staticmethod
    def l2(p1, p2):
        """
        :param p1: 2x1 matrix
        :param p2: 2x1 matrix
        :return: euclidean distance
        """
        return ((p1[0][0] - p2[0][0]) ** 2 + (p1[1][0] - p2[1][0]) ** 2) ** 0.5

    @staticmethod
    def get_xy(sec):
        return np.array([[sec[X]], [sec[Y]]])

    def sin(self, p1, p2):
        """
        :param p1: 2x1 matrix
        :param p2: 2x1 matrix
        :return: scalar sinus
        """
        return (p2[1] - p1[1])[0] / self.l2(p1, p2)

    def cos(self, p1, p2):
        """
        :param p1: 2x1 matrix
        :param p2: 2x1 matrix
        :return: scalar cosinus
        """
        return (p2[0] - p1[0])[0] / self.l2(p1, p2)

    def get_ball_pos(self):
        conn = MySqlConnection()
        conn.execute_query(Sqls.GET_BALL_POS_DATA % self.game_id)
        for each in conn.get_cursor():
            self.ball_pos_data.append(each)

    def calculate_influence(self, p1, p2, speed, ref_p, inf_part, covariance_matrix, base=False):
        mean = self.calculate_muit(p1, p2, speed)
        if base:
            ref_p = mean
        distance_form_mean = ref_p - mean

        exp_part = math.exp(-0.5 * reduce(np.matmul, [distance_form_mean.T,
                                                      np.linalg.inv(covariance_matrix),
                                                      distance_form_mean]))

        return inf_part * exp_part

    def calculate_covit(self, p1, p2, speed, ball_pos):
        """
        :param p1: 2x1 matrix
        :param p2: 2x1 matrix
        :param speed: scalar
        :param ball_pos: 2d numpy array
        :return: covariance matrix 2x2
        """
        rotit = self.calculate_rotit(p1, p2)
        sit = self.calculate_sit(p1, speed, ball_pos=ball_pos)

        return reduce(np.matmul, [rotit, sit, sit, rotit.T])

    def calculate_rotit(self, p1, p2):
        """
        :param p1: 2x1 matrix
        :param p2: 2x1 matrix
        :return: rotation matrix
        """
        #  trying mirrored sin value since y is increasing downward in pitch coordinate system.
        return np.array([[self.cos(p1, p2), -self.sin(p1, p2)], [self.sin(p1, p2), self.cos(p1, p2)]])

    def calculate_sit(self, p1, speed, ball_pos):
        """
        :param p1: 2x1 matrix
        :param speed: scalar
        :param ball_pos: 2x1 matrix
        :return: scaling matrix
        """
        rit_score = self.rit(p1, ball_pos)
        scaling = rit_score * (speed ** 2 / 169.0)

        return np.array([[(rit_score + scaling) / 2, 0], [0, (rit_score - scaling) / 2]])

    def rit(self, pp, bp):
        return math.sin((self.l2(pp, bp) * (2 / 110) - 1) ** 2) * 10 + 4

    def calculate_muit(self, p1, p2, speed):
        """
        :param p1: 2x1 matrix
        :param p2: 2x1 matrix
        :param speed: scalar
        :return:
        """
        x_factor = self.cos(p1, p2)
        y_factor = self.sin(p1, p2)

        x_part = p1[0] + ((speed * x_factor) / 2)
        y_part = p1[1] + ((speed * y_factor) / 2)
        return np.array([x_part, y_part])

    def add_defensive_players(self, ball_pos1):
        ball_speed = 10
        for c, player in enumerate(self.def_players_data_at_sec):
            p1 = self.get_xy(player[0])
            p2 = self.get_xy(player[1])
            speed = self.l2(p1, p2)

            diff = self.l2(p1, ball_pos1)
            exxp = (-math.sin((diff * (2.0 / 110) - 1) ** 7) - math.sin(diff * (2.0 / 110) - 1)) / 1.75

            sec1_speed_x = speed / 4.0 * self.cos(p1, p2) + ball_speed * exxp * self.cos(
                ball_pos1,
                p1)
            sec1_speed_y = speed / 4.0 * self.sin(p1, p2) + ball_speed * exxp * self.sin(
                ball_pos1,
                p1)

            p2[0][0] += ball_speed * self.cos(ball_pos1, p1) * exxp
            p2[1][0] += ball_speed * self.sin(ball_pos1, p1) * exxp
            speed = (sec1_speed_x ** 2 + sec1_speed_y ** 2) ** 0.5

            covariance_matrix = self.calculate_covit(p1, p2, speed, ball_pos1)
            inf_part = 1.0 / (((2 * math.pi) ** 2 * np.linalg.det(covariance_matrix)) ** 0.5)
            base_inf = self.calculate_influence(p1, p2, speed, p1, inf_part, covariance_matrix, base=True)

            for i, x in enumerate(self.X_RANGE):
                for j, y in enumerate(self.Y_RANGE):
                    self.results[i][j] -= self.calculate_influence(p1, p2, speed, np.array([[x], [y]]), inf_part,
                                                                   covariance_matrix) / base_inf

    def normalize(self):
        maxx = np.max(self.results)
        minn = np.min(self.results)

        for i, x in enumerate(self.X_RANGE):
            for j, y in enumerate(self.Y_RANGE):
                self.results[i][j] = (self.results[i][j] - minn) / (maxx - minn)

    def show(self, ball_pos=None, transpose=False):
        results = np.array(self.results)
        if transpose:
            results = results.transpose()

        fig, axs = plt.subplots(1, 1)
        im = axs.imshow(results, extent=[0, 110, 0, 68])
        plt.colorbar(im)
        self.scat_players_and_ball(ball_pos)
        # plt.savefig(name)
        plt.show()

    def scat_players_and_ball(self, ball_pos):
        for player in self.def_players_data_at_sec:
            x, y = self.get_xy(player[0])
            plt.scatter(x, y, c='red', s=60)
        for player in self.off_players_data_at_sec:
            x, y = self.get_xy(player[0])
            plt.scatter(x, y, c='green', s=60)

        plt.scatter(ball_pos[0][0], ball_pos[1][0], c='white', s=75)

    @staticmethod
    def to21_array(tpl):
        return np.array([[tpl[0]], [tpl[1]]])

    def initialize(self):
        for i, x in enumerate(self.X_RANGE):
            for j, y in enumerate(self.Y_RANGE):
                self.results[i][j] = 0

    def show_all(self, off_data, def_data):
        self.def_players_data_at_sec = def_data
        self.off_players_data_at_sec = off_data

        ball_pos1 = PassProbability.to21_array(self.ball_pos_dict[self.off_players_data_at_sec[0][0][SEC]])

        self.initialize()
        self.add_defensive_players(ball_pos1=ball_pos1)
        self.normalize()
        # self.results = np.array(self.results).transpose()

        # self.show(ball_pos=ball_pos1)
        return self.results


# pv = PassProbability(2, 3, 68322)
# plx = 25
# ply = 50
# p1 = np.array([[plx], [ply]])
# p2 = np.array([[plx+2], [ply+2]])
# speed = pv.l2(p1, p2)
# ball_pos = np.array([[plx], [10]])
# diff = pv.l2(p1, ball_pos)
# exxp = (-math.sin((diff * (2.0 / 110) - 1) ** 7) - math.sin(diff * (2.0 / 110) - 1)) / 1.75
#
# sec1_speed_x = speed / 4.0 * pv.cos(p1, p2) + 15 * exxp * pv.cos(
#     ball_pos,
#     p1)
# sec1_speed_y = speed / 4.0 * pv.sin(p1, p2) + 15 * exxp * pv.sin(
#     ball_pos,
#     p1)
#
# p2[0][0] += BALL_SPEED * pv.cos(ball_pos, p1) * exxp
# p2[1][0] += BALL_SPEED * pv.sin(ball_pos, p1) * exxp
# speed = (sec1_speed_x ** 2 + sec1_speed_y ** 2) ** 0.5
#
# covariance_matrix = pv.calculate_covit(p1, p2, speed, ball_pos)
# inf_part = 1.0 / (((2 * math.pi) ** 2 * np.linalg.det(covariance_matrix)) ** 0.5)
# base_inf = pv.calculate_influence(p1, p2, speed, p1, inf_part, covariance_matrix)
#
# for i, x in enumerate(pv.X_RANGE):
#     for j, y in enumerate(pv.Y_RANGE):
#         pv.results[i][j] -= pv.calculate_influence(p1, p2, speed, np.array([[x], [y]]), inf_part,
#                                                    covariance_matrix) / base_inf
#
# fig, axs = plt.subplots(1, 1)
# im = axs.imshow(pv.results.T, extent=[0, 110, 0, 68])
# plt.scatter([plx], [ply], c='red', s=60)
# plt.scatter([plx], [15], c='white', s=60)
# plt.plot([plx, plx+2], [ply, ply+2])
# plt.arrow(25, 15, 0, 15, edgecolor='white', facecolor='white', shape='full',
#           length_includes_head=True, head_width=2, head_length=2)
# plt.show()
