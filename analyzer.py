# -*- coding: utf-8 -*-

import traceback

from data_loaders.get_a_game_data import GameData
from data_loaders.get_a_game_ball_data import BallData
from analyzers.closeness_analyzer import ClosenessAnalyzer
from analyzers.marking_analyzer import MarkingAnalyzer
from analyzers.pass_analyzer import PassAnalyzer
from plotters.matrix_plotter import MatrixPlotter
from validator import Validator
from match_statistics import MatchStatistics


class Analyzer:
    def __init__(self, matrix_plotter):
        self.closeness_analyzer = None
        self.marking_analyzer = None
        self.pass_analyzer = None
        self.game_data_collector = None
        self.ball_data_collector = None
        self.game_data = None
        self.db_data = None
        self.team_ids = []
        self.pass_team_matrices = []
        self.matrix_plotter = matrix_plotter
        self.sec_splitter_index = 0
        self.data_arrays = {}
        self.printed = False

    def calculate_closeness(self):
        self.game_data_collector.get_data(file_name=None)
        self.closeness_analyzer = ClosenessAnalyzer(self.game_data_collector.db_data)
        self.game_data = self.game_data_collector.game_data
        self.db_data = self.game_data_collector.db_data
        self.team_ids = [self.game_data_collector.db_data[0].id, self.game_data_collector.db_data[1].id]

    def calculate_passes(self):
        self.ball_data_collector.get_data(file_name=None)
        self.match_collectors_player_names()
        for team in self.ball_data_collector.db_data:
            self.pass_analyzer = PassAnalyzer(self.ball_data_collector.ball_data, team)
            self.pass_analyzer.analyze_data()
            self.pass_team_matrices.append(self.pass_analyzer.teammate_matrix)

    def calculate_marking(self):
        self.marking_analyzer = MarkingAnalyzer(self.game_data_collector.db_data)

    def match_collectors_player_names(self):
        self.ball_data_collector.db_data[0].set_player_names(self.db_data[0].get_player_names())
        self.ball_data_collector.db_data[0].set_jersey_numbers(self.db_data[0].get_jersey_numbers())
        self.ball_data_collector.db_data[1].set_player_names(self.db_data[1].get_player_names())
        self.ball_data_collector.db_data[1].set_jersey_numbers(self.db_data[1].get_jersey_numbers())

    def calculate_average_team_length(self, ms):
        half = 1
        minn = 0
        sec = 0

        while True:
            sec_data = self.get_a_sec_data(self.game_data, half, minn, sec)

            if not sec_data:
                ms.not_sec_data += 1
                if minn/half<45:
                    sec += 1
                    if sec == 60:
                        sec = 0
                        minn += 1
                    continue
                half += 1

                if half == 3:
                    sec_max = minn*60 + sec
                    break

                sec = 0
                minn = 45
                continue

            # At least 22 players exist and a team has ball
            if sec_data[0][-2] != 0 and len(sec_data) >= 22:
                ms.secs_secs[sec_data[0][1]] = {'dists': None, 'p2p': {}}

                my_dist1, my_dist1_w, my_dist2, my_dist2_w = self.calculate_dist(sec_data, ms=ms, my_way=True)
                norm_dist1_x, norm_dist1_y, norm_dist2_x, norm_dist2_y = self.calculate_dist(sec_data)

                ms.secs_secs[sec_data[0][1]]['dists'] = [my_dist1, my_dist1_w,
                                                         my_dist2, my_dist2_w,
                                                         norm_dist1_x, norm_dist1_y,
                                                         norm_dist2_x, norm_dist2_y]
                ms.secs_secs_as_list.append([my_dist1, my_dist1_w,
                                             my_dist2, my_dist2_w,
                                             norm_dist1_x, norm_dist1_y,
                                             norm_dist2_x, norm_dist2_y])

                ms.inc_my_totdist(my_dist1, my_dist1_w, my_dist2, my_dist2_w)
                ms.inc_norm_totdist(norm_dist1_x, norm_dist1_y, norm_dist2_x, norm_dist2_y)

                if not self.printed:
                    # my_dist_home = self.calculate_dist(sec_data, my_way=True)[0]
                    # my_dist_away = self.calculate_dist(sec_data, my_way=True)[1]
                    # norm_dist_home = self.calculate_dist(sec_data, my_way=False)[0]
                    # norm_dist_away = self.calculate_dist(sec_data, my_way=False)[1]
                    # dp = DistPlotter(sec_data,
                    #                  self.team_ids,
                    #                  normal_dists=[norm_dist_home, norm_dist_away],
                    #                  my_dists=[my_dist_home, my_dist_away])

                    # dp.plot_pitch()
                    # dp.put_players_on_pitch()
                    # dp.plot_norm_dist()
                    # dp.show_dist_stats()
                    # dp.show()
                    self.printed = True

            if sec_data[0][-2] == 0:
                state = 0
            elif sec_data[0][-2] == self.team_ids[0]:
                state = 1
            else:
                state = 2
            ms.increment_hasball_secs(state)

            if len(sec_data) <= 22:
                ms.player_data_not_sufficient[str(half)+str(minn)+str(sec)] = len(sec_data)

            sec += 1
            if sec == 60:
                minn += 1
                sec = 0

        ms.set_cohesive_matrices(self.data_arrays['team1_total_def'],
                                 self.data_arrays['team1_total_off'],
                                 self.game_data_collector.db_data[0].get_player_names())

        return ms.get_return_info()

    def get_a_sec_data(self, game_data, half, minn, sec):
        return_value = None
        for i in range(self.sec_splitter_index, len(game_data)):
            if game_data[i][1] == half*10000 + minn*100 + sec:
                continue
            else:
                return_value = game_data[self.sec_splitter_index:i]
                self.sec_splitter_index = i
                break
        return return_value

    def get_a_sec_data_only(self, game_data, sec_key):
        return filter(lambda x: x[1] == sec_key, game_data)

    def my_calculation(self, team_data, team_players, ind, hasball=False, ms=None, sec=None):
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
                    ms.secs_secs[sec]['p2p'][get_name(p1)][get_name(p1)] = 0
                    continue
                dist_meter = ((p1[5] - p2[5]) ** 2 + (p1[6] - p2[6]) ** 2) ** 0.5
                p1_ind = team_players.index(get_name(p1))
                p2_ind = team_players.index(get_name(p2))
                norm_total_dist += dist_meter

                factor = self.data_arrays['team%d_total_off' % (ind + 1)][p1_ind][p2_ind]
                if not hasball:
                    factor = self.data_arrays['team%d_total_def' % (ind + 1)][p1_ind][p2_ind]

                if ms:
                    if not ms.secs_secs[sec]['p2p'].get(get_name(p1)):
                        ms.secs_secs[sec]['p2p'][get_name(p1)] = {}
                    ms.secs_secs[sec]['p2p'][get_name(p1)][get_name(p2)] = dist_meter * factor
                total_dist += dist_meter * factor
                total_count += 1
        # return norm_total_dist / float(total_count), total_dist / float(total_count)
        return norm_total_dist**0.5, total_dist**0.5

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
        hasball = self.team_ids[0] == sec_data[0][0]
        sec = sec_data[0][1]

        team_players1 = self.game_data_collector.db_data[0].get_player_names()
        team_data1 = filter(lambda x: x[0] == self.team_ids[0], sec_data)
        team_players2 = self.game_data_collector.db_data[1].get_player_names()
        team_data2 = filter(lambda x: x[0] == self.team_ids[1], sec_data)

        if my_way:
            avg_dist1, avg_dist1_w = self.my_calculation(team_data1, team_players1, 0, hasball, ms=ms, sec=sec)
            avg_dist2, avg_dist2_w = self.my_calculation(team_data2, team_players2, 1, not hasball, ms=ms, sec=sec)
            return avg_dist1, avg_dist1_w, avg_dist2, avg_dist2_w
        else:
            avg_dist1_x, avg_dist1_y = self.norm_calculation(team_data1, 0)
            avg_dist2_x, avg_dist2_y = self.norm_calculation(team_data2, 1)
            return avg_dist1_x, avg_dist1_y, avg_dist2_x, avg_dist2_y

    def create_2d_arrays(self):
        import numpy as np

        def normalize(team_matrix):
            return team_matrix / float(np.amax(team_matrix))

        for i in range(2):
            self.data_arrays['team%d_closeness' % (i+1)] = MatrixPlotter.dict_to_matrix(
                self.closeness_analyzer.team_matrices[i], self.game_data_collector.db_data[i].get_player_names(), True)
            self.data_arrays['team%d_passes' % (i + 1)] = MatrixPlotter.dict_to_matrix(
                self.pass_team_matrices[i], self.game_data_collector.db_data[i].get_player_names(), True)
            self.data_arrays['team%d_marking' % (i + 1)] = MatrixPlotter.dict_to_matrix(
                self.marking_analyzer.team_matrices[i], self.game_data_collector.db_data[i].get_player_names(), True)
            self.data_arrays['team%d_total_off' % (i + 1)] = \
                normalize(self.data_arrays['team%d_closeness' % (i+1)] + self.data_arrays['team%d_passes' % (i + 1)])
            self.data_arrays['team%d_total_def' % (i + 1)] = \
                normalize(self.data_arrays['team%d_closeness' % (i+1)] + self.data_arrays['team%d_marking' % (i + 1)])


if __name__ == "__main__":
    import time
    import pprint
    print '#' * 49
    print '#' * 49
    pprint.pprint('la sikik')
    print '#' * 49
    print '#' * 49
    a = time.time()
    validator = Validator()
    validator.get_games()
    print "Get game:", time.time() - a; a = time.time()
    games = validator.return_games()
    matrix_plotter = MatrixPlotter()
    mine = 0
    nrml = 0

    just_home = True
    midway = True

    for cnt, game in enumerate(games):
        analyzer = Analyzer(matrix_plotter)
        ms = MatchStatistics(analyzer)

        a = time.time()
        analyzer.game_data_collector = GameData(get_type='Multiple', game=games[game])
        print "game data:", time.time() - a; a = time.time()
        analyzer.ball_data_collector = BallData(get_type='Multiple', game=games[game])
        print "ball data:", time.time() - a; a = time.time()
        analyzer.calculate_closeness()
        print "closeness:", time.time() - a; a = time.time()
        analyzer.calculate_passes()
        print "passes:", time.time() - a; a = time.time()
        analyzer.calculate_marking()
        print "marking:", time.time() - a; a = time.time()

        # importance_list = []
        # for i in range(0, 2):
        #     players = analyzer.game_data_collector.db_data[i].get_player_names()
        #     cls = MatrixPlotter.dict_to_matrix(analyzer.closeness_analyzer.team_matrices[i], players)
        #     mrk = MatrixPlotter.dict_to_matrix(analyzer.marking_analyzer.team_matrices[i], players)
        #     pss = MatrixPlotter.dict_to_matrix(analyzer.pass_team_matrices[i], players)
        #
        #     importance = cls + mrk + pss
        #     importance_list.append([sum(importance[c]) for c, p in enumerate(players)])

        analyzer.matrix_plotter.set_closeness_matrix(analyzer.closeness_analyzer.team_matrices[0])
        analyzer.matrix_plotter.set_keys()
        analyzer.matrix_plotter.set_pass_matrix(analyzer.pass_team_matrices[0])
        analyzer.matrix_plotter.set_marking_matrix(analyzer.marking_analyzer.team_matrices[0])
        # analyzer.matrix_plotter.plot()
        # scatter = PitchScatter(analyzer.game_data_collector.db_data)
        # analyzer.matrix_plotter.set_scatter(scatter)
        # analyzer.matrix_plotter.plot_scatter(importance_list)

        try:
            analyzer.create_2d_arrays()
            mh, mh_w, ma, ma_w, nh_x, nh_y, na_x, na_y = analyzer.calculate_average_team_length(ms)
            ms.print_ms()
            ms.dist_plotter()
            ms.scenario_plotter(just_home, midway)
            print "######################################"
            print "######################################"
            print 'Names', '\t\t', games[game]['home']['name'], '\t\t', games[game]['away']['name']
            print 'Scores', '\t\t', games[game]['score'][0], '\t\t\t\t', games[game]['score'][2]
            print 'Norm. Dist', '\t', nh_x, nh_y, '\t\t', na_x, na_y
            print 'My Dist', '\t', mh_w, '\t\t', ma_w
            print 'Rates', '\t', mh_w/ma_w, '\t\t', nh_x/na_x
            print "######################################"
            print "######################################"

            winner_home = 1
            if int(games[game]['score'].split('-')[0]) < int(games[game]['score'].split('-')[1]):
                winner_home = 0
                
            if (mh_w/ma_w < nh_x/na_x and winner_home) \
                    or (mh_w/ma_w > nh_x/na_x and not winner_home):
                mine += 1
            else:
                nrml += 1

            break

        except Exception, e:
            print "PROBLEM IN", games[game]['home']['name'], games[game]['away']['name'], "GAME."
            print e
            traceback.print_exc()

    print mine, nrml


###################################################################################################
###################################################################################################
# TODO marking iyi bir markaj durumunu plot et
###################################################################################################
###################################################################################################