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
X = 5
Y = 6
SPEED = 10
NAME = 11
SIZE_X = 100
SIZE_Y = 60


class PitchValue:
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

    def calculate_influence(self, p1, p2, speed, ref_p=None, ball_pos=None, factor=1.0, base=False):
        """
        :param p1: 2x1 matrix
        :param p2: 2x1 matrix
        :param speed: scalar
        :param ref_p: 2x1 matrix
        :param ball_pos: 2x1 matrix
        :param factor: scalar
        :return:
        """
        if base:
            ref_p = p1

        covariance_matrix = self.calculate_covit(p1, p2, speed, ball_pos=ball_pos)
        mean = self.calculate_muit(p1, p2, speed)

        distance_form_mean = ref_p - mean

        exp_part = math.exp(-0.5 * reduce(np.matmul, [distance_form_mean.T,
                                                      np.linalg.inv(covariance_matrix),
                                                      distance_form_mean]))

        inf_part = 1.0 / (((2 * math.pi) ** 2 * np.linalg.det(covariance_matrix)) ** 0.5)

        if factor < 1:
            return min(((inf_part * exp_part) ** factor) / (3 / factor), 0.075)
        return inf_part * exp_part

    def calculate_player_influence(self, p1, p2, speed, ref_p, covariance_matrix, inf_part, base=False):
        if base:
            ref_p = p1

        mean = self.calculate_muit(p1, p2, speed)
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
        return min((self.l2(pp, bp) ** 3) / 7500.0 + 4, 10)*2

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

    def initialize(self, is_away):
        ball_x = self.ball_pos_dict[self.off_players_data_at_sec[0][0][SEC]][0]
        for i, x in enumerate(self.X_RANGE):
            for j, y in enumerate(self.Y_RANGE):
                self.results[i][j] = (x - ball_x) / 65 * (-1 if is_away else 1)

    def add_players_goal_ball(self, ball_pos1, ball_pos2, is_away):
        for pindex, player in enumerate(self.def_players_data_at_sec):
            p1 = self.get_xy(player[0])
            p2 = self.get_xy(player[1])
            speed = self.l2(p1, p2)

            goal_p1 = np.array([[110], [34]])
            goal_p2 = np.array([[109.9], [34]])
            if is_away:
                goal_p1 = np.array([[0], [34]])
                goal_p2 = np.array([[0.1], [34]])
            goal_speed = 0.1

            ball_speed = self.l2(ball_pos1, ball_pos2)

            covariance_matrix = self.calculate_covit(p1, p2, speed, ball_pos1)
            inf_part = 1.0 / (((2 * math.pi) ** 2 * np.linalg.det(covariance_matrix)) ** 0.5)
            base_inf = self.calculate_player_influence(p1, p2, speed, p1, covariance_matrix, inf_part, base=True)
            base_goal_inf = self.calculate_influence(goal_p1, goal_p2, goal_speed, goal_p1, ball_pos1, base=True)
            base_ball_inf = self.calculate_influence(ball_pos1, ball_pos2, ball_speed, ball_pos1, ball_pos1, base=True)

            for i, x in enumerate(self.X_RANGE):
                for j, y in enumerate(self.Y_RANGE):
                    self.results[i][j] += self.calculate_player_influence(p1, p2,
                                                                          speed, np.array([[x], [y]]),
                                                                          covariance_matrix, inf_part) / base_inf
                    if not pindex:
                        self.results[i][j] += self.calculate_influence(goal_p1, goal_p2, goal_speed,
                                                                       np.array([[x], [y]]),
                                                                       ball_pos1) / base_goal_inf
                        self.results[i][j] += self.calculate_influence(ball_pos1, ball_pos2, ball_speed,
                                                                       np.array([[x], [y]]),
                                                                       ball_pos1) / base_ball_inf

    def normalize(self):
        maxx = np.max(self.results)
        minn = np.min(self.results)

        for i, x in enumerate(self.X_RANGE):
            for j, y in enumerate(self.Y_RANGE):
                self.results[i][j] = (self.results[i][j] - minn) / (maxx - minn)

    def show(self, prep_results=None, ball_pos=None):
        # if prep_results is not None:
        #     results = np.array(prep_results).transpose()
        # else:
        #     results = np.array(self.results).transpose()
        fig, axs = plt.subplots(1, 1)
        im = axs.imshow(self.results, extent=[0, 110, 0, 68])
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

    def show_all(self, off_data, def_data, is_away):
        self.def_players_data_at_sec = def_data
        self.off_players_data_at_sec = off_data

        ball_pos1 = PitchValue.to21_array(self.ball_pos_dict[self.off_players_data_at_sec[0][0][SEC]])
        ball_pos2 = PitchValue.to21_array(self.ball_pos_dict[self.off_players_data_at_sec[0][1][SEC]])

        self.initialize(is_away)
        self.add_players_goal_ball(ball_pos1=ball_pos1, ball_pos2=ball_pos2, is_away=is_away)
        self.normalize()
        # self.results = np.array(self.results).transpose()

        # self.show(ball_pos=ball_pos1)
        return self.results

# pv = PitchValue(2, 3, 68322)
# p1 = np.array([[25], [20]])
# p2 = np.array([[27], [22]])
# ball_pos = np.array([[25], [10]])
# base_inf = pv.calculate_influence(p1, p2, pv.l2(p1, p2), ref_p=p1, ball_pos=ball_pos, factor=0.3)
# print "Base INF:", base_inf
# for i, x in enumerate(pv.X_RANGE):
#     for j, y in enumerate(pv.Y_RANGE):
#         ref_p = np.array([[x], [y]])
#         inf = pv.calculate_influence(p1, p2, pv.l2(p1, p2), ref_p=ref_p, ball_pos=ball_pos, factor=0.3)
#         # print x, "(%d)"%i,y, "(%d)"%j, inf, inf/base_inf
#         pv.results[i][j] = inf / base_inf
#
# fig, axs = plt.subplots(1, 1)
# im = axs.imshow(pv.results.T, extent=[0, 110, 0, 68])
# plt.show()
