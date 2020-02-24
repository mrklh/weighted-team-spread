# -*- coding: utf-8 -*-

import pickle
import pprint
import traceback
from asynchat import simple_producer

import numpy as np

from data_loaders.get_a_game_data import GameData
from data_loaders.get_a_game_ball_data import BallData
# from pitch_value import PitchValue
from analyzers.closeness_analyzer import ClosenessAnalyzer
from analyzers.marking_analyzer import MarkingAnalyzer
from analyzers.pass_analyzer import PassAnalyzer
from plotters.matrix_plotter import MatrixPlotter
from validator import Validator
from match_statistics import MatchStatistics
from data_loaders.pickle_loader import PickleLoader
from plotters.pitch_plotter import PitchPlotter
from data_loaders.db_data_collector import DbDataCollector
from commons import Commons
from usable_high_value_area import UHVA

import pprint

import matplotlib.pyplot as plt

S_TEAM_ID = 2
S_JERSEY = 3
S_HALF = 9
S_MIN = 10
S_SEC = 11
S_END_X = 16
S_END_Y = 17
S_HOME_LEFT = -4

P_TEAM_ID = 0
P_X = 5
P_Y = 6
P_JERSEY = 7
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
        self.weight_matrices = {}
        self.printed = False

        self.team_names = []

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
        # secs = map(lambda x: analyzer.get_a_sec_data_only(analyzer.game_data, x), range(10000, 14824) + range(24500, 29609))
        # import pickle
        #
        # with open('pickles/pitch_value_data2.pkl', 'wb+') as f:
        #     pickle.dump(secs, f)
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
        # secs = map(lambda x: analyzer.get_a_sec_data_only(analyzer.ball_data, x), range(10000, 14824) + range(24500, 29609))
        # import pickle
        #
        # with open('pickles/ball_data2.pkl', 'wb+') as f:
        #     pickle.dump(secs, f)

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
            if sec_data[0][-3] != 0 and len(sec_data) >= 22:
            # if len(sec_data) >= 22:
                ms.secs_secs[sec_data[0][1]] = {'dists': None, 'p2p': {}, 'dist_matrix': {}}

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

            if sec_data[0][-3] == 0:
                state = 0
            elif sec_data[0][-3] == self.team_ids[0]:
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
        ms.set_cohesive_matrices(self.weight_matrices['team1_total_def'],
                                 self.weight_matrices['team1_total_off'],
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
                norm_total_dist += dist_meter**2

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
                total_dist += (dist_meter * factor)**2
                total_count += 1
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
            self.weight_matrices['team%d_closeness' % (i+1)] = self.closeness_matrices[i]
            self.weight_matrices['team%d_passes' % (i + 1)] = self.pass_matrices[i]
            self.weight_matrices['team%d_marking' % (i + 1)] = self.marking_matrices[i]
            self.weight_matrices['team%d_total_off' % (i + 1)] = \
                normalize(self.weight_matrices['team%d_closeness' % (i+1)] + self.weight_matrices['team%d_passes' % (i + 1)])
            self.weight_matrices['team%d_total_def' % (i + 1)] = \
                normalize(self.weight_matrices['team%d_closeness' % (i+1)] + self.weight_matrices['team%d_marking' % (i + 1)])

    def save_weights(self):
        pickled_data = {}
        pickled_data['closeness'] = analyzer.closeness_matrices
        pickled_data['pass'] = analyzer.pass_matrices
        pickled_data['marking'] = analyzer.marking_matrices

        # self.pickle_loader.dump_data(pickled_data)


if __name__ == "__main__":
    import time

    start = time.time()
    validator = Validator()
    validator.get_games_from_db()
    print "Get game\t:", "%.2f secs" % (time.time() - start)
    games = validator.return_games()
    game_events_by_type = {}
    teams_events = None

    mine = 0
    nrml = 0

    just_home = True
    midway = True

    for cnt, game in enumerate(games):
        print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        print '%d of %s games' % (cnt + 1, len(games.keys()))
        try:

            sprint_dict = {}

            print '#' * 49
            print '#' * 49
            pprint.pprint(games[game])
            print '#' * 49
            print '#' * 49
            analyzer = Analyzer(games[game], game_events_by_type)
            ms = MatchStatistics(analyzer)

            analyzer.game_data_collector = GameData(get_type='Multiple', game=games[game])
            analyzer.ball_data_collector = BallData(get_type='Multiple', game=games[game])

            analyzer.closeness_matrices, analyzer.pass_matrices, analyzer.marking_matrices = analyzer.get_matrices_pickled()
            if not analyzer.pickled:
                analyzer.calculate_closeness()
                analyzer.calculate_passes()
                analyzer.calculate_marking()
                # analyzer.save_weights()
            else:
                # analyzer.calculate_closeness()
                analyzer.calculate_passes()

            # pv = PitchValue(analyzer)
            # pv.plot_pitch_with_values()

            # importance_list = []
            # for i in range(0, 2):
            #     players = analyzer.game_data_collector.db_data[i].get_player_names()
            #     cls = Commons.dict_to_matrix(analyzer.closeness_analyzer.p2p_dicts[i], players)
            #     mrk = Commons.dict_to_matrix(analyzer.marking_analyzer.p2p_dicts[i], players)
            #     pss = Commons.dict_to_matrix(analyzer.pass_p2p_dicts[i], players)
            #
            #     importance = cls + mrk + pss
            #     importance_list.append([sum(importance[c]) for c, p in enumerate(players)])
            from data_loaders.sqls import Sqls
            from data_loaders.mine_sql_connection import MySqlConnection

            conn = MySqlConnection()
            conn.execute_query(Sqls.GET_FIRST_ELEVEN % game)
            first_eleven = conn.get_cursor().fetchall()
            analyzer.team_names = [x[1] for x in first_eleven]
            analyzer.nums = [x[0] for x in first_eleven]
            analyzer.jersey = {x[1]: x[0] for x in first_eleven}

            # if not cnt:
            #     team_names = Commons.bjk_kon
            # else:
            #     team_names = Commons.bjk_bsk

            if games[game]['home']['id'] == 3:
                index = 0
            else:
                index = 1
            analyzer.matrix_plotter.set_closeness_matrix(analyzer.closeness_analyzer.p2p_dicts[index])
            analyzer.matrix_plotter.set_keys(analyzer.team_names)
            analyzer.matrix_plotter.set_pass_matrix(analyzer.pass_analyzer.p2p_dicts[index])
            analyzer.matrix_plotter.set_marking_matrix(analyzer.marking_analyzer.p2p_dicts[index])
            print '#' * 49
            print '#' * 49
            pprint.pprint(games[game]['home']['id'])
            pprint.pprint(Commons.teams[games[game]['home']['id']])
            print '#' * 49
            print '#' * 49
            analyzer.matrix_plotter.game_name = Commons.teams[games[game]['home']['id']] + "_" + \
                                                Commons.teams[games[game]['away']['id']]

            # analyzer.matrix_plotter.plot_pass_network(analyzer)
            # analyzer.matrix_plotter.plot(id=cnt)
            # scatter = PitchScatter(analyzer.game_data_collector.db_data)
            # analyzer.matrix_plotter.set_scatter(scatter)
            # analyzer.matrix_plotter.plot_scatter(importance_list)
            # continue
            analyzer.create_2d_arrays_of_cohesions()
            mh, mh_w, ma, ma_w, nh_x, nh_y, na_x, na_y = analyzer.calculate_average_team_length(ms)
            ms.print_ms()
            ms.set_sec_keys_of_ball_data()
            # ms.dist_plotter()
            # ms.scenario_plotter(just_home, midway)
            # ms.trace_game_events()
            # teams_events = ms.split_game_events_to_teams()

            conn.execute_query(Sqls.GET_SPRINT_DATA % game)
            sprint_data = conn.get_cursor().fetchall()

            uhva = UHVA(game, analyzer.teams[0].id, analyzer.teams[1].id)

            print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            print "LEN SPRINT", len(sprint_data)
            print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            for spc, sprint in enumerate(sprint_data):
                if not spc % 10:
                    print "%d sprint calculated." % spc
                sprint_time = 10000*sprint[S_HALF] + 100*sprint[S_MIN] + sprint[S_SEC]

                if sprint[S_SEC] == 59:
                    next_time = 10000*sprint[S_HALF] + 100*sprint[S_MIN] + 1
                    next_min = sprint[S_MIN] + 1
                    next_sec = 0
                else:
                    next_time = 10000*sprint[S_HALF] + 100*sprint[S_MIN] + sprint[S_SEC] + 1
                    next_min = sprint[S_MIN]
                    next_sec = sprint[S_SEC] + 1

                if not uhva.pv.ball_pos_dict.get(sprint_time) or not uhva.pv.ball_pos_dict.get(next_time):
                    continue

                team_index = 1
                ts_index = 3
                if analyzer.teams[0].id == sprint[S_TEAM_ID]:
                    team_index = 0
                    ts_index = 1
                if ms.secs_secs.get(sprint_time) and ms.secs_secs.get(next_time):
                    # get weighted team spread of the attacking team at corresponding second
                    spread = ms.secs_secs[sprint_time]['dists'][ts_index]

                    # get pos data of teams at corresponding second
                    sec_data1 = analyzer.get_a_sec_data_only(analyzer.game_data, sprint_time)
                    # get pos data for defending team at next of the corresponding second
                    sec_data2 = analyzer.get_a_sec_data_only(analyzer.game_data, next_time)


                    # get off_team_data for two seconds
                    off_players = [x[P_NAME] for x in filter(lambda x: x[P_TEAM_ID] == sprint[S_TEAM_ID], sec_data1)]
                    continuous_data = [sec_data1] + [sec_data2]
                    off_data = list([[filter(lambda x: x[P_NAME] == z and x[P_TEAM_ID] == sprint[S_TEAM_ID], y)[0]
                                 for y in continuous_data] for z in off_players])

                    # get def_team_data for two seconds
                    def_players = [x[P_NAME] for x in filter(lambda x: x[P_TEAM_ID] != sprint[S_TEAM_ID], sec_data1)]
                    def_data = list([[filter(lambda x: x[P_NAME] == z and x[P_TEAM_ID] != sprint[S_TEAM_ID], y)[0]
                                 for y in continuous_data] for z in def_players])

                    # get ref_p
                    try:
                        sprinting_player = filter(lambda x: x[P_JERSEY] == sprint[S_JERSEY]
                                                            and x[P_TEAM_ID] == sprint[S_TEAM_ID], sec_data2)[0]
                    except:
                        continue
                    ref_p = [sprinting_player[P_X],  sprinting_player[P_Y]]

                    uhva.set_frob(ms.secs_secs.get(sprint_time)['dists'][ts_index])
                    uhva.mean()
                    uhva_val = uhva.calculate_UHVA(sec=sprint_time, off_data=off_data, def_data=def_data, ref_p=ref_p)
                    if not sprint_dict.get(str(game) + "_" + str(sprint[S_TEAM_ID]) + "_" + str(sprint[S_JERSEY])):
                        sprint_dict[str(game) + "_" + str(sprint[S_TEAM_ID]) + "_" + str(sprint[S_JERSEY])] = []

                    sprint_dict[str(game) + "_" + str(sprint[S_TEAM_ID]) + "_" + str(sprint[S_JERSEY])].append(uhva_val)
                    # uhva.show(sec=time)

            Reporter(games[game], nh_x, nh_y, na_x, na_y, mh_w, ma_w)

            winner_home = 1
            with open("pickles/%d_sprints.pkl" % game, "wb") as f:
                pickle.dump(sprint_dict, f)

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

    with open('pickles/teams_events_real_spread.pkl', 'wb+') as f:
        pickle.dump(teams_events, f)

    import pickle
    with open('pickles/bag_of_events_real_spread.pkl', 'wb+') as f:
        pickle.dump(analyzer.events_by_type, f)

    for typ in game_events_by_type:
        home_events = filter(lambda x: x['event'][1] == 3, game_events_by_type[typ])
        for event in home_events:
            plt.plot(event['flow'])
        plt.title('%d event %d count' % (typ, len(home_events)))
        plt.show()

    print mine, nrml


###################################################################################################
###################################################################################################
# TODO marking iyi bir markaj durumunu plot et
###################################################################################################
###################################################################################################