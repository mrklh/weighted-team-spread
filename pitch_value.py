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
        self.results = np.zeros((50, 50))
        with open('pickles/pitch_value_data.pkl', 'rb+') as f:
            self.data = pickle.load(f)

        self.get_ball_pos()
        self.ball_pos_dict = {x[0]: (x[4], x[5]) for x in filter(lambda x: x[0] in [x[0][1] for x in self.data], self.ball_pos_data)}

    @staticmethod
    def l2(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    @staticmethod
    def get_xy(sec):
        return sec[X], sec[Y]
    
    def sin(self, p1, p2):
        return (p2[1] - p1[1])/self.l2(p1, p2)

    def cos(self, p1, p2):
        return (p2[0] - p1[0])/self.l2(p1, p2)

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
        exp_part = math.exp(-0.5*(np.matmul(exp_part1.transpose(), exp_part2)))
        inf_part1 = 1.0/(((2*math.pi)**2*np.linalg.det(covit))**0.5)

        if factor < 1:
            return min(((inf_part1*exp_part)**factor)/(3/factor), 0.035)
        return inf_part1*exp_part

    def calculate_covit(self, sec1, sec2):
        rotit = self.calculate_rotit(self.get_xy(sec1), self.get_xy(sec2))
        sit = self.calculate_sit(sec1)
        return reduce(np.matmul, [rotit, sit, sit, np.linalg.inv(rotit)])

    def calculate_rotit(self, p1, p2):
        return np.array([[self.cos(p1, p2), -self.sin(p1, p2)], [self.sin(p1, p2), self.cos(p1, p2)]])
    
    def calculate_sit(self, sec):
        rit_score = self.rit((sec[X], sec[Y]), self.ball_pos_dict[sec[SEC]])
        scaling = rit_score * (sec[SPEED]**2/169)

        return np.array([[(rit_score+scaling)/2, 0], [0, (rit_score-scaling)/2]])

    def rit(self, pp, bp):
        return min((self.l2(pp, bp)**3)/1000.0 + 4, 10)

    def calculate_muit(self, sec1, sec2):
        return np.array([self.get_xy(sec1)[0] + (sec2[X] - sec1[X])/2, self.get_xy(sec1)[1] + (sec2[Y] - sec1[Y])/2])

    def add_defensive_players(self):
        def_players = [x[NAME] for x in filter(lambda x: x[0] == 101, pv.data[0])]
        def_players_data_at_0 = [[filter(lambda x: x[NAME] == z, y)[0] for y in pv.data[:2]] for z in def_players]

        for i, x in enumerate(self.X_RANGE):
            for j, y in enumerate(self.Y_RANGE):
                self.results[i][j] = 0.30

        for player in def_players_data_at_0:
            sec1 = player[0]
            sec2 = player[1]
            pprint.pprint('%s POSITION %.3f-%.3f' % (sec1[NAME], self.get_xy(sec1)[0], self.get_xy(sec1)[1]))

            for i, x in enumerate(self.X_RANGE):
                for j, y in enumerate(self.Y_RANGE):
                    self.results[i][j] -= self.calculate_influence(sec1, sec2, ref_p=[x, y], factor=0.5)

    def add_goal(self):
        sec1 = [0] * 11
        sec1[Y] = 34
        sec1[X] = 110
        sec1[SPEED] = 0.1
        sec1[SEC] = self.data[0][0][SEC]
        sec2 = [0] * 11
        sec2[Y] = 34
        sec2[X] = 109.9
        sec2[SPEED] = 0
        sec2[SEC] = self.data[1][0][SEC]
        print "Add goal Position %.3f-%.3f" % (sec1[X], sec1[Y])

        for i, x in enumerate(self.X_RANGE):
            for j, y in enumerate(self.Y_RANGE):
                self.results[i][j] += self.calculate_influence(sec1, sec2, ref_p=[x, y], factor=0.1)

    def add_ball(self):
        sec1 = [0] * 11
        sec1[Y] = self.ball_pos_dict[self.data[0][0][SEC]][0]
        sec1[X] = self.ball_pos_dict[self.data[0][0][SEC]][1]
        sec1[SPEED] = 0.1
        sec1[SEC] = self.data[0][0][SEC]
        sec2 = [0] * 11
        sec2[Y] = self.ball_pos_dict[self.data[1][0][SEC]][0]
        sec2[X] = self.ball_pos_dict[self.data[1][0][SEC]][1]
        sec2[SPEED] = 0
        sec2[SEC] = self.data[1][0][SEC]
        print "Add ball Position %.3f-%.3f" % (sec1[X], sec1[Y])

        for i, x in enumerate(self.X_RANGE):
            for j, y in enumerate(self.Y_RANGE):
                self.results[i][j] += self.calculate_influence(sec1, sec2, ref_p=[x, y], factor=0.1)

    def normalize(self):
        maxx = np.max(self.results)
        minn = np.min(self.results)

        for i, x in enumerate(self.X_RANGE):
            for j, y in enumerate(self.Y_RANGE):
                self.results[i][j] = (self.results[i][j] - minn) / (maxx - minn)

    def show(self):
        results = np.array(self.results).transpose()
        fig, axs = plt.subplots(1, 1)
        im = axs.imshow(results, extent=[0, 110, 0, 68])
        plt.colorbar(im)
        # plt.savefig('defensive_influence.pdf')
        plt.show()

    def show_all(self):
        self.add_defensive_players()
        self.add_goal()
        self.add_ball()
        self.normalize()
        self.show()


pv = PitchValue()
pv.show_all()
