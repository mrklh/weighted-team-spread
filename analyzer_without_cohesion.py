# -*- coding: utf-8 -*-

import pickle
import os
import sys
import traceback
import psutil
import pprint
import datetime

import matplotlib.pyplot as plt

from data_loaders.get_a_game_data import GameData
from plotters.matrix_plotter import MatrixPlotter
from validator import Validator
from match_statistics import MatchStatistics
from data_loaders.pickle_loader import PickleLoader
from usable_high_value_area_with_range import UHVA
from data_loaders.sqls import Sqls
from data_loaders.mine_sql_connection import MySqlConnection
from custom_utils.memory_profile import get_size

S_TEAM_ID = 2
S_JERSEY = 3
S_HALF = 9
S_MIN = 10
S_SEC = 11
S_END_SEC = 13
S_BEGIN_X = 14
S_BEGIN_Y = 15
S_END_X = 16
S_END_Y = 17

P_TEAM_ID = 0
P_X = 5
P_Y = 6
P_JERSEY = 7
HASBALL_TEAM_ID = 8
P_NAME = 11


class Reporter:
    def __init__(self, game, nh_x, nh_y, na_x, na_y, mh_w, ma_w):
        print "######################################"
        print "######################################"
        print game['id']
        print 'Names', '\t\t', game['home']['name'], '\t\t', game['away']['name']
        print 'Scores', '\t\t', game['score'][0], '\t\t\t\t', game['score'][2]
        print 'Norm. Dist', '\t', nh_x, nh_y, '\t\t', na_x, na_y
        print 'My Dist', '\t', mh_w, '\t\t', ma_w
        print 'Rates', '\t', mh_w / ma_w, '\t\t', nh_x / na_x
        print "######################################"
        print "######################################"


class Analyzer:
    def __init__(self, game_info):
        # Analyzers
        self.closeness_analyzer = None
        self.marking_analyzer = None
        self.pass_analyzer = None

        # Analyzed matrices
        self.closeness_matrices = []
        self.pass_matrices = []
        self.marking_matrices = []

        # Data collectors
        self.game_data_collector = None
        self.ball_data_collector = None

        # Home and away team data holders
        self.teams = None
        self.team_ids = []

        # Position data holder
        self.game_data = None
        self.ball_data = None

        self.pass_matrices = []

        # Helper fields
        self.pickled = False
        self.matrix_plotter = MatrixPlotter()
        self.sec_splitter_index = 0
        self.weight_matrices = {}
        self.printed = False

        self.team_names = []

        self.pickle_loader = PickleLoader("game_data/" + str(game_info['id']) + ".pkl")

    class RunFuncWithTimer(object):
        def __init__(self, tag):
            self.tag = tag
            self.pref = "\t"
            if len(self.tag) <= 6:
                self.pref = "\t\t"

        def __call__(self, func):
            def func_wrapper(parent):
                start = time.time()
                func(parent)
                end = time.time()
                print self.tag, '%s: %.2f secs' % (self.pref, end - start)

            return func_wrapper

    def print_size(self):
        print "Size of analyzer: %.2f MB" % (get_size(self)/2.**20)

    def get_matrices_pickled(self):
        data = self.pickle_loader.return_data()
        if not data:
            return None, None
        else:
            self.pickled = True
            return data['closeness'], data['pass'], data['marking']

    @RunFuncWithTimer('Closeness')
    def calculate_closeness(self):
        self.game_data_collector.get_data(file_name=None)
        self.game_data = self.game_data_collector.game_data

        self.teams = self.game_data_collector.db_data
        self.team_ids = [self.teams[0].id, self.teams[1].id]
        self.set_keys()

    def set_keys(self):
        self.home_keys = self.teams[0].get_player_names()
        self.away_keys = self.teams[1].get_player_names()

    def calculate_average_team_length(self, ms):
        half = 1
        minn = 0
        sec = 0

        while True:
            sec_data = self.get_a_sec_data(self.game_data, half, minn, sec)

            if not sec_data:
                ms.not_sec_data += 1
                if minn / half < 45:
                    sec += 1
                    if sec == 60:
                        sec = 0
                        minn += 1
                    continue
                half += 1

                if half == 3:
                    sec_max = minn * 60 + sec
                    break

                sec = 0
                minn = 45
                continue

            # At least 22 players exist and a team has ball
            if sec_data[0][-3] != 0 and len(sec_data) >= 22:
                ms.secs_secs[sec_data[0][1]] = 1

    def get_a_sec_data(self, game_data, half, minn, sec):
        return_value = None
        for i in range(self.sec_splitter_index, len(game_data)):
            if game_data[i][1] == half * 10000 + minn * 100 + sec:
                continue
            else:
                return_value = game_data[self.sec_splitter_index:i]
                self.sec_splitter_index = i
                break
        return return_value

    def get_a_sec_data_only(self, game_data, sec_key):
        # print sec_key
        sec_data = filter(lambda x: x[1] == sec_key, game_data)
        return [list(x) for x in sec_data]

    def my_calculation(self, team_data, team_players, ind, hasball=False, ms=None, sec=None):
        '''
        Team spread value is calculated here.

        :param team_data:
        :param team_players:
        :param ind:
        :param hasball:
        :param ms:
        :param sec:
        :return:
        '''

        def get_name(p_data):
            return p_data[-1] or 'Unknown Player'

        total_dist = 0
        norm_total_dist = 0
        total_count = 0
        for p1 in team_data:
            for p2 in team_data:
                if p1[-1] == p2[-1]:
                    if not ms.secs_secs[sec]['p2p'].get(get_name(p1)):
                        ms.secs_secs[sec]['p2p'][get_name(p1)] = {}
                    if not ms.secs_secs[sec]['dist_matrix'].get(get_name(p1)):
                        ms.secs_secs[sec]['dist_matrix'][get_name(p1)] = {}

                    ms.secs_secs[sec]['p2p'][get_name(p1)][get_name(p1)] = 0
                    ms.secs_secs[sec]['dist_matrix'][get_name(p1)][get_name(p1)] = 0
                    continue
                dist_meter = ((p1[5] - p2[5]) ** 2 + (p1[6] - p2[6]) ** 2) ** 0.5
                p1_ind = team_players.index(get_name(p1))
                p2_ind = team_players.index(get_name(p2))
                norm_total_dist += dist_meter ** 2

                factor = self.weight_matrices['team%d_total_off' % (ind + 1)][p1_ind][p2_ind]
                if not hasball:
                    factor = self.weight_matrices['team%d_total_def' % (ind + 1)][p1_ind][p2_ind]

                if ms:
                    if not ms.secs_secs[sec]['p2p'].get(get_name(p1)):
                        ms.secs_secs[sec]['p2p'][get_name(p1)] = {}
                    if not ms.secs_secs[sec]['dist_matrix'].get(get_name(p1)):
                        ms.secs_secs[sec]['dist_matrix'][get_name(p1)] = {}

                    ms.secs_secs[sec]['p2p'][get_name(p1)][get_name(p2)] = dist_meter * factor
                    ms.secs_secs[sec]['dist_matrix'][get_name(p1)][get_name(p2)] = dist_meter
                total_dist += (dist_meter * factor) ** 2
                total_count += 1
        return norm_total_dist ** 0.5, total_dist ** 0.5

    def norm_calculation(self, team_data, ind):
        if ind == 0:
            sec_min_x_ind = [x[5] for x in team_data].index(sorted([x[5] for x in team_data])[1])
            sec_max_x_ind = [x[5] for x in team_data].index(sorted([x[5] for x in team_data])[-1])

            sec_min_y_ind = [x[6] for x in team_data].index(sorted([x[6] for x in team_data])[1])
            sec_max_y_ind = [x[6] for x in team_data].index(sorted([x[6] for x in team_data])[-1])
        else:
            sec_min_x_ind = [x[5] for x in team_data].index(sorted([x[5] for x in team_data])[-2])
            sec_max_x_ind = [x[5] for x in team_data].index(sorted([x[5] for x in team_data])[0])

            sec_min_y_ind = [x[6] for x in team_data].index(sorted([x[6] for x in team_data])[-2])
            sec_max_y_ind = [x[6] for x in team_data].index(sorted([x[6] for x in team_data])[0])
        sec_min_x = team_data[sec_min_x_ind]
        sec_max_x = team_data[sec_max_x_ind]

        sec_min_y = team_data[sec_min_y_ind]
        sec_max_y = team_data[sec_max_y_ind]

        return abs(sec_min_x[5] - sec_max_x[5]), abs(sec_min_y[6] - sec_max_y[6])

    def calculate_dist(self, sec_data, ms=None, my_way=False):
        hasball = self.team_ids[0] == sec_data[0][-3]
        sec = sec_data[0][1]

        team_players1 = self.teams[0].get_player_names()
        team_data1 = filter(lambda x: x[0] == self.team_ids[0], sec_data)
        team_players2 = self.teams[1].get_player_names()
        team_data2 = filter(lambda x: x[0] == self.team_ids[1], sec_data)

        if my_way:
            avg_dist1, avg_dist1_w = self.my_calculation(team_data1, team_players1, 0, hasball, ms=ms, sec=sec)
            avg_dist2, avg_dist2_w = self.my_calculation(team_data2, team_players2, 1, not hasball, ms=ms, sec=sec)
            return avg_dist1, avg_dist1_w, avg_dist2, avg_dist2_w
        else:
            avg_dist1_x, avg_dist1_y = self.norm_calculation(team_data1, 0)
            avg_dist2_x, avg_dist2_y = self.norm_calculation(team_data2, 1)
            return avg_dist1_x, avg_dist1_y, avg_dist2_x, avg_dist2_y

    def create_2d_arrays_of_cohesions(self):
        import numpy as np

        def normalize(team_matrix):
            return team_matrix / float(np.amax(team_matrix))

        for i in range(2):
            self.weight_matrices['team%d_closeness' % (i + 1)] = self.closeness_matrices[i]
            self.weight_matrices['team%d_passes' % (i + 1)] = self.pass_matrices[i]
            self.weight_matrices['team%d_marking' % (i + 1)] = self.marking_matrices[i]
            self.weight_matrices['team%d_total_off' % (i + 1)] = \
                normalize(self.weight_matrices['team%d_closeness' % (i + 1)] + self.weight_matrices[
                    'team%d_passes' % (i + 1)])
            self.weight_matrices['team%d_total_def' % (i + 1)] = \
                normalize(self.weight_matrices['team%d_closeness' % (i + 1)] + self.weight_matrices[
                    'team%d_marking' % (i + 1)])

    def save_weights(self):
        pickled_data = {}
        pickled_data['closeness'] = analyzer.closeness_matrices
        pickled_data['pass'] = analyzer.pass_matrices
        pickled_data['marking'] = analyzer.marking_matrices

        # self.pickle_loader.dump_data(pickled_data)

    def show_pitch(self, off_data, def_data):
        fig, axs = plt.subplots(1, 1, figsize=(14, 6))
        plt.axis('off')

        img = plt.imread("resources/pitch.png")
        axs.imshow(img, extent=[0, 110, 0, 68], alpha=0.6)

        for player in def_data:
            x, y = player[0][P_X], player[0][P_Y]
            axs.scatter(x, y, c='red', s=100)

        for player in off_data:
            x, y = player[0][P_X], player[0][P_Y]
            axs.scatter(x, y, c='green', s=100)

        plt.show()
        return


if __name__ == "__main__":
    import time

    start = time.time()
    validator = Validator()
    validator.get_games_from_db()
    print "Get game\t:", "%.2f secs" % (time.time() - start)
    games = validator.return_games()

    mine = 0
    nrml = 0

    just_home = True
    midway = True

    pid = os.getpid()
    py = psutil.Process(pid)

    processed = 0

    for cnt, game in enumerate(games):
        if processed == 70:
            break

        if os.path.exists("plots/%d" % game):
            print "[GAME-%d] Already processed. Passing..." % game
            continue

        processed += 1
        print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        memoryUse = py.memory_info()[0] / 2. ** 30
        print "Current memory usage: %.2f GB" % memoryUse
        print '%d of %s games' % (cnt + 1, len(games.keys()))
        try:
            sprint_dict = {}

            print '#' * 49
            print '#' * 49
            pprint.pprint(games[game])
            print '#' * 49
            print '#' * 49
            analyzer = Analyzer(games[game])
            ms = MatchStatistics(analyzer)
            analyzer.print_size()

            pickled_data = analyzer.pickle_loader.return_data()
            if pickled_data:
                analyzer.game_data, analyzer.teams, ms.secs_secs = pickled_data
            else:
                analyzer.game_data_collector = GameData(get_type='Multiple', game=games[game])
                analyzer.calculate_closeness()
                analyzer.calculate_average_team_length(ms)
                analyzer.pickle_loader.dump_data((analyzer.game_data, analyzer.teams, ms.secs_secs))

            conn = MySqlConnection()
            conn.execute_query(Sqls.GET_SPRINT_DATA % game)
            sprint_data = conn.get_cursor().fetchall()

            uhva = UHVA(game, analyzer.teams[0].id, analyzer.teams[1].id)

            print "[GAME-%d] Sprint count: %d" % (game, len(sprint_data))
            missing_data = 0
            defender_sprint = 0
            ball_position = 0
            start = time.time()
            for spc, sprint in enumerate(sprint_data):
                if spc and not spc % 15:
                    print "[GAME-%d] Iteration [%d]" % (game, spc)
                sprint_time = 10000 * sprint[S_HALF] + 100 * sprint[S_MIN] + sprint[S_SEC]

                if sprint[S_SEC] == 59:
                    next_time = 10000 * sprint[S_HALF] + 100 * sprint[S_MIN] + 1
                    next_min = sprint[S_MIN] + 1
                    next_sec = 0
                else:
                    next_time = 10000 * sprint[S_HALF] + 100 * sprint[S_MIN] + sprint[S_SEC] + 1
                    next_min = sprint[S_MIN]
                    next_sec = sprint[S_SEC] + 1

                # check ball pos exist
                if not uhva.pv.ball_pos_dict.get(sprint_time) or not uhva.pv.ball_pos_dict.get(next_time):
                    ball_position += 1
                    continue

                team_index = 1
                if analyzer.teams[0].id == sprint[S_TEAM_ID]:
                    team_index = 0
                if ms.secs_secs.get(sprint_time) and ms.secs_secs.get(next_time):
                    # get pos data of teams at corresponding second
                    sec_data1 = analyzer.get_a_sec_data_only(analyzer.game_data, sprint_time)
                    # get pos data for defending team at next of the corresponding second
                    sec_data2 = analyzer.get_a_sec_data_only(analyzer.game_data, next_time)

                    # check sprinter is in the attacking team
                    if sec_data1[0][HASBALL_TEAM_ID] != sprint[S_TEAM_ID]:
                        defender_sprint += 1
                        continue

                    # get off_team_data for two seconds
                    off_players = [x[P_NAME] for x in filter(lambda x: x[P_TEAM_ID] == sprint[S_TEAM_ID], sec_data1)]
                    continuous_data = [sec_data1] + [sec_data2]
                    off_data = list([[filter(lambda x: x[P_NAME] == z and x[P_TEAM_ID] == sprint[S_TEAM_ID], y)[0]
                                      for y in continuous_data] for z in off_players])

                    # get def_team_data for two seconds
                    def_players = [x[P_NAME] for x in filter(lambda x: x[P_TEAM_ID] != sprint[S_TEAM_ID], sec_data1)]
                    def_data = list([[filter(lambda x: x[P_NAME] == z and x[P_TEAM_ID] != sprint[S_TEAM_ID], y)[0]
                                      for y in continuous_data] for z in def_players])

                    value = uhva.calculate_UHVA(is_away=team_index, off_data=off_data, def_data=def_data,
                                                attack_team_id=sprint[S_TEAM_ID], sprint=sprint)

                    with open("pickles/uhvas/%d_%d" % (game, spc), "wb") as f:
                        pickle.dump({'off_data': off_data, 'def_data': def_data, 'uhva': uhva.UHVA, 'value': value,
                                     'sprint': sprint}, f)

                    if not sprint_dict.get(str(game) + "_" + str(sprint[S_TEAM_ID]) + "_" + str(sprint[S_JERSEY])):
                        sprint_dict[str(game) + "_" + str(sprint[S_TEAM_ID]) + "_" + str(sprint[S_JERSEY])] = []

                    sprint_dict[str(game) + "_" + str(sprint[S_TEAM_ID]) + "_" + str(sprint[S_JERSEY])].append(value)
                    uhva.show(spc=spc, value=value, sprint=sprint, game=game)
                else:
                    missing_data += 1

            duration = (time.time() - start)
            print "[GAME-%d] Total duration: %s" % (game, (str(datetime.timedelta(seconds=duration))))
            print "[GAME-%d] Missing: %d, Defender Sprint: %d, Ball Prob. %d" % (game,
                                                                                 missing_data,
                                                                                 defender_sprint,
                                                                                 ball_position)
            analyzer.print_size()
            del uhva
            del ms
            del analyzer
        except Exception, e:
            print "PROBLEM IN", games[game]['home']['name'], games[game]['away']['name'], "GAME."
            print e
            traceback.print_exc()

###################################################################################################
###################################################################################################
# TODO marking iyi bir markaj durumunu plot et
###################################################################################################
###################################################################################################
