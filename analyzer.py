# -*- coding: utf-8 -*-

import pprint
import traceback

from data_loaders.get_a_game_data import GameData
from data_loaders.get_a_game_ball_data import BallData
from analyzers.closeness_analyzer import ClosenessAnalyzer
from analyzers.marking_analyzer import MarkingAnalyzer
from analyzers.pass_analyzer import PassAnalyzer
from plotters.matrix_plotter import MatrixPlotter
from validator import Validator
from match_statistics import MatchStatistics
from data_loaders.pickle_loader import PickleLoader
from data_loaders.db_data_collector import DbDataCollector
from commons import Commons

import matplotlib.pyplot as plt

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
    def __init__(self, game_info, game_events_by_type):
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
        self.events = []

        # Position data holder
        self.game_data = None
        self.ball_data = None

        self.pass_matrices = []

        # All games events list
        self.events_by_type = game_events_by_type

        # Helper fields
        self.pickled = False
        self.matrix_plotter = MatrixPlotter()
        self.sec_splitter_index = 0
        self.data_arrays = {}
        self.printed = False
        self.pickle_loader = PickleLoader("matrix_%s_%s" % (Commons.get_team_abb(game_info['home']['name']),
                                                            Commons.get_team_abb(game_info['away']['name'])))

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

    def get_matrices_pickled(self):
        data = self.pickle_loader.return_data()
        if not data:
            return None, None, None
        else:
            self.pickled = True
            return data['closeness'], data['pass'], data['marking']

    @RunFuncWithTimer('Closeness')
    def calculate_closeness(self):
        self.game_data_collector.get_data(file_name=None)
        self.game_data = self.game_data_collector.game_data
        self.teams = self.game_data_collector.db_data
        self.events = self.game_data_collector.events
        self.team_ids = [self.teams[0].id, self.teams[1].id]
        self.set_keys()
        if analyzer.pickled:
            return
        self.closeness_analyzer = ClosenessAnalyzer(self)

    @RunFuncWithTimer('Passes')
    def calculate_passes(self):
        self.ball_data_collector.get_data(file_name=None)
        self.ball_data = self.ball_data_collector.ball_data
        if analyzer.pickled:
            return
        self.pass_analyzer = PassAnalyzer(self)

    @RunFuncWithTimer('Marking')
    def calculate_marking(self):
        self.marking_analyzer = MarkingAnalyzer(self)

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

                # my_dist1 = frobenius
                # my_dist1_w = w. frobenius

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

        ms.secs_secs.keys()
        ms.set_cohesive_matrices(self.data_arrays['team1_total_def'],
                                 self.data_arrays['team1_total_off'],
                                 self.teams[0].get_player_names())

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
        return norm_total_dist / float(total_count), total_dist / float(total_count)

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

    def create_2d_arrays(self):
        import numpy as np

        def normalize(team_matrix):
            return team_matrix / float(np.amax(team_matrix))

        for i in range(2):
            self.data_arrays['team%d_closeness' % (i+1)] = self.closeness_matrices[i]
            self.data_arrays['team%d_passes' % (i + 1)] = self.pass_matrices[i]
            self.data_arrays['team%d_marking' % (i + 1)] = self.marking_matrices[i]
            self.data_arrays['team%d_total_off' % (i + 1)] = \
                normalize(self.data_arrays['team%d_closeness' % (i+1)] + self.data_arrays['team%d_passes' % (i + 1)])
            self.data_arrays['team%d_total_def' % (i + 1)] = \
                normalize(self.data_arrays['team%d_closeness' % (i+1)] + self.data_arrays['team%d_marking' % (i + 1)])

    def save_weights(self):
        pickled_data = {}
        pickled_data['closeness'] = analyzer.closeness_matrices
        pickled_data['pass'] = analyzer.pass_matrices
        pickled_data['marking'] = analyzer.marking_matrices

        self.pickle_loader.dump_data(pickled_data)

if __name__ == "__main__":
    import time

    start = time.time()
    validator = Validator()
    validator.get_games_from_db()
    print "Get game\t:", "%.2f secs" % (time.time() - start)
    games = validator.return_games()
    game_events_by_type = {}

    mine = 0
    nrml = 0

    just_home = True
    midway = True

    for cnt, game in enumerate(games):
        analyzer = Analyzer(games[game], game_events_by_type)
        ms = MatchStatistics(analyzer)

        analyzer.game_data_collector = GameData(get_type='Multiple', game=games[game])
        analyzer.ball_data_collector = BallData(get_type='Multiple', game=games[game])

        analyzer.closeness_matrices, analyzer.pass_matrices, analyzer.marking_matrices = analyzer.get_matrices_pickled()
        if not analyzer.pickled:
            analyzer.calculate_closeness()
            analyzer.calculate_passes()
            analyzer.calculate_marking()
            analyzer.save_weights()
        else:
            analyzer.calculate_closeness()
            analyzer.calculate_passes()

        # importance_list = []
        # for i in range(0, 2):
        #     players = analyzer.game_data_collector.db_data[i].get_player_names()
        #     cls = Commons.dict_to_matrix(analyzer.closeness_analyzer.p2p_dicts[i], players)
        #     mrk = Commons.dict_to_matrix(analyzer.marking_analyzer.p2p_dicts[i], players)
        #     pss = Commons.dict_to_matrix(analyzer.pass_p2p_dicts[i], players)
        #
        #     importance = cls + mrk + pss
        #     importance_list.append([sum(importance[c]) for c, p in enumerate(players)])

        analyzer.matrix_plotter.set_closeness_matrix(analyzer.closeness_matrices[0])
        analyzer.matrix_plotter.set_keys()
        analyzer.matrix_plotter.set_pass_matrix(analyzer.pass_matrices[0])
        analyzer.matrix_plotter.set_marking_matrix(analyzer.marking_matrices[0])
        # analyzer.matrix_plotter.plot()
        # scatter = PitchScatter(analyzer.game_data_collector.db_data)
        # analyzer.matrix_plotter.set_scatter(scatter)
        # analyzer.matrix_plotter.plot_scatter(importance_list)

        try:
            analyzer.create_2d_arrays()
            mh, mh_w, ma, ma_w, nh_x, nh_y, na_x, na_y = analyzer.calculate_average_team_length(ms)
            ms.print_ms()
            # ms.dist_plotter()
            # ms.scenario_plotter(just_home, midway)
            ms.trace_game_events()

            Reporter(games[game], nh_x, nh_y, na_x, na_y, mh_w, ma_w)

            winner_home = 1
            if int(games[game]['score'].split('-')[0]) < int(games[game]['score'].split('-')[1]):
                winner_home = 0
                
            if (mh_w/ma_w < nh_x/na_x and winner_home) \
                    or (mh_w/ma_w > nh_x/na_x and not winner_home):
                mine += 1
            else:
                nrml += 1

        except Exception, e:
            print "PROBLEM IN", games[game]['home']['name'], games[game]['away']['name'], "GAME."
            print e
            traceback.print_exc()

    # for type in game_events_by_type:
    #     home_events = filter(lambda x: x['event'][1] == 3, game_events_by_type[type])
    #     for event in home_events:
    #         plt.plot(event['flow'])
    #     plt.title('%d event %d count' % (type, len(home_events)))
    #     plt.show()
    #
    # print mine, nrml


###################################################################################################
###################################################################################################
# TODO marking iyi bir markaj durumunu plot et
###################################################################################################
###################################################################################################