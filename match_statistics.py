# -*- coding: utf-8 -*-

import pprint
import numpy as np

from scipy.stats.stats import pearsonr

from plotters.scenario_plotter import ScenarioPlotter
from plotters.dist_time_plotter import DistTimePlotter
from commons import Commons

class MatchStatistics:
    def __init__(self, analyzer):
        self.analyzer = analyzer

        self.my_total_dist_h = 0
        self.my_total_dist_h_w = 0
        self.my_total_dist_a = 0
        self.my_total_dist_a_w = 0
        self.norm_total_dist_h_x = 0
        self.norm_total_dist_h_y = 0
        self.norm_total_dist_a_x = 0
        self.norm_total_dist_a_y = 0

        self.nobody_hasball = 0
        self.home_hasball = 0
        self.away_hasball = 0

        self.player_data_not_sufficient = {}
        self.not_sec_data = 0

        self.secs_secs = {}
        self.secs_secs_as_list = []

        self.def_cohesive_matrix = None
        self.off_cohesive_matrix = None
        self.names = None

        self.sec_keys_of_ball_data = None

        self.closest = 1000
        self.closest_scores = (0, 0)
        self.farthest = 0
        self.farthest_scores = (0, 0)
        self.closest_sec = 0
        self.farthest_sec = 0

    def set_sec_keys_of_ball_data(self):
        self.sec_keys_of_ball_data = [x[0] for x in self.analyzer.ball_data]

    def increment_hasball_secs(self, state):
        '''
        Increments seconds info

        :param state: 0: nobody 1: home 2: away has ball
        '''
        if state == 1:
            self.home_hasball += 1
        elif state == 2:
            self.away_hasball += 1
        else:
            self.nobody_hasball += 1

    def inc_my_totdist(self, dist_h, dist_h_w, dist_a, dist_a_w):
        self.my_total_dist_h += dist_h
        self.my_total_dist_h_w += dist_h_w

        self.my_total_dist_a += dist_a
        self.my_total_dist_a_w += dist_a_w

    def inc_norm_totdist(self, dist_h_x, dist_h_y, dist_a_x, dist_a_y):
        self.norm_total_dist_h_x += dist_h_x
        self.norm_total_dist_h_y += dist_h_y

        self.norm_total_dist_a_x += dist_a_x
        self.norm_total_dist_a_y += dist_a_y

    def get_total_sec(self):
        return self.home_hasball + self.away_hasball

    def get_return_info(self):
        return self.my_total_dist_h/self.get_total_sec(), \
               self.my_total_dist_h_w/self.get_total_sec(), \
               self.my_total_dist_a/self.get_total_sec(), \
               self.my_total_dist_a_w/self.get_total_sec(), \
               self.norm_total_dist_h_x/self.get_total_sec(), \
               self.norm_total_dist_h_y/self.get_total_sec(), \
               self.norm_total_dist_a_x/self.get_total_sec(), \
               self.norm_total_dist_a_y/self.get_total_sec()

    def set_cohesive_matrices(self, def_matrix, off_matrix, names):
        self.def_cohesive_matrix = def_matrix
        self.off_cohesive_matrix = off_matrix
        self.names = names

    def print_ms(self):
        print "////////////////////////////////////////////////////////"
        print "////////////////////////////////////////////////////////"
        print "nobody_hasball", self.nobody_hasball
        print "home_hasball", self.home_hasball
        print "away_hasball", self.away_hasball
        print "not_sec_data", self.not_sec_data
        print "player_data_not_sufficient", len(self.player_data_not_sufficient.keys())
        print "correlation bw frob-mine-h", pearsonr([x[0] for x in self.secs_secs_as_list],
                                                     [x[1] for x in self.secs_secs_as_list])
        print "correlation bw my2-n2", pearsonr([x[1] for x in self.secs_secs_as_list],
                                                [x[3] for x in self.secs_secs_as_list])
        print "////////////////////////////////////////////////////////"
        print "////////////////////////////////////////////////////////"

    def find_beginning_of_attacking_transaction(self, event_time):
        sec_keys = sorted(self.secs_secs.keys())

        init_hasball_team = 0
        time_list = []
        time = event_time
        gap = 0

        while True:
            # 3 saniyelik bir boşluk var burada keseceğiz.
            if gap == 4:
                # print 'returns at %d because gap of 4' % time
                return list(reversed(time_list))

            # Veri bozulduysa return
            if time not in sec_keys:
                # print 'Found a gap at %d' % time,
                gap += 1
                time = Commons.decrease_time(time)
                continue
            else:
                gap = 0

            hasball_team = self.find_who_has_ball(time)

            # Olay anında topun kimde olduğunu bulalım.
            if not init_hasball_team and hasball_team:
                init_hasball_team = hasball_team
            elif not init_hasball_team and not hasball_team:
                gap += 1

            # Topa sahip takım değiştiyse return
            if hasball_team != init_hasball_team:
                # print 'returns at %d because hasball changed' % time
                return list(reversed(time_list))

            time_list.append(time)

            # 1 saniye geri gidelim.
            time = Commons.decrease_time(time)

            # Devrenin başına geldiysek return
            if time < 0:
                # print 'returns at %d because we are at beginning of half' % time
                return list(reversed(time_list))

    def find_who_has_ball(self, time):
        try:
            data = self.analyzer.ball_data[self.sec_keys_of_ball_data.index(time)]
            return data[2]
        except:
            return -1

    def interpolate_event_flow(self, time_list, event_flow):
        new_event_flow = []
        for c, time in enumerate(time_list[:-1]):
            new_event_flow.append(event_flow[c])
            if time != Commons.decrease_time(time_list[c+1]):
                gap = 0

                inter_time = Commons.increase_time(time)
                while inter_time != time_list[c+1]:
                    gap += 1
                    inter_time = Commons.increase_time(inter_time)

                wfrob_diff = event_flow[c+1] - event_flow[c]
                for i in range(1, gap+1):
                    new_event_flow.append(event_flow[c] + (wfrob_diff/(gap+1))*i)

        new_event_flow.append(event_flow[-1])
        
        return new_event_flow

    # def interpolate_dist_flow(self, time_list, dist_flow):
    #     new_dist_flow = None
    #     for i in range(11):
    #         for j in range(11):
    #             flow = [x[i][j] for x in dist_flow]
    #             new_flow = self.interpolate_event_flow(time_list, flow)
    #
    #             # Ilk loopta saniye sayisi kadar 11x11 matrix olusturalim.
    #             if not i and not j:
    #                 new_dist_flow = [np.zeros((11, 11))] * len(new_flow)
    #
    #             for c, each in enumerate(new_flow):
    #                 new_dist_flow[c][i][j] = each
    #
    #     return new_dist_flow

    def prepare_dist_flow(self, time_list, is_home):
        player_names = self.analyzer.teams[0 if is_home else 1].get_player_names()

        dist_flow = [Commons.dict_to_matrix(self.secs_secs[x]['dist_matrix'], player_names, is_normalize=False)
                     for x in time_list]
        # return self.interpolate_dist_flow(time_list, dist_flow)
        return dist_flow

    def prepare_event_flow(self, time_list, is_home):
        index = 1 if is_home else 3
        base = self.secs_secs[time_list[0]]['dists'][index]

        # uncomment to calculate real team spread
        event_flow = [self.secs_secs[x]['dists'][index] for x in time_list]

        # event_flow = [self.secs_secs[x]['dists'][index] - base for x in time_list]
        return self.interpolate_event_flow(time_list, event_flow)

    def trace_game_events(self):
        '''
        Traces all events and finds if it has continuous series of seconds of x
        It adds event actioned by a team with same id, adds -id for opposite team.
        '''
        for event in self.analyzer.events:
            is_home = event[1] == self.analyzer.teams[0].id
            time_list = self.find_beginning_of_attacking_transaction(event[0])

            if len(time_list) < 5:
                continue

            event_flow_commit = self.prepare_event_flow(time_list, is_home)
            dist_flow_commit = self.prepare_dist_flow(time_list, is_home)

            event_flow_suffer = self.prepare_event_flow(time_list, not is_home)
            dist_flow_suffer = self.prepare_dist_flow(time_list, not is_home)

            if not self.analyzer.events_by_type.get(event[-1]):
                self.analyzer.events_by_type[event[-1]] = []

            if not self.analyzer.events_by_type.get(-event[-1]):
                self.analyzer.events_by_type[-event[-1]] = []

            # append format:
            # for team actioned event.
            # {
            # event     : event
            # flow      : event_flow
            # dist_flow : distance_flow
            # closeness : closeness matrix
            # pass      : pass matrix
            # }
            # for opposite team actioned event
            # {
            # event     : -event
            # flow      : event_flow
            # dist_flow : distance_flow
            # closeness : closeness matrix
            # pass      : pass matrix
            # }
            self.analyzer.events_by_type[event[-1]].append({
                'event': event,
                'flow': event_flow_commit,
                'dist_flow': dist_flow_commit,
                'closeness': self.analyzer.weight_matrices['team%d_closeness' % (1 if is_home else 2)],
                'passes': self.analyzer.weight_matrices['team%d_passes' % (1 if is_home else 2)]
            })

            event = (event[0],
                     self.analyzer.teams[0].id if not is_home else self.analyzer.teams[1].id,
                     event[2],
                     -event[3])
            self.analyzer.events_by_type[event[-1]].append({
                'event': event,
                'flow': event_flow_suffer,
                'dist_flow': dist_flow_suffer,
                'closeness': self.analyzer.weight_matrices['team%d_closeness' % (1 if not is_home else 2)],
                'passes': self.analyzer.weight_matrices['team%d_passes' % (1 if not is_home else 2)]
            })

    def split_game_events_to_teams(self):
        teams = {}
        for key in self.analyzer.events_by_type:
            for event in self.analyzer.events_by_type[key]:
                if not teams.get(event['event'][1]):
                    teams[event['event'][1]] = {key: []}
                if not teams[event['event'][1]].get(key):
                    teams[event['event'][1]][key] = []

                teams[event['event'][1]][key].append(event)

        return teams

    def scenario_plotter(self, just_home, midway):
        sec_list = []
        sec_keys = sorted(self.secs_secs.keys())

        rng = [x for x in sec_keys[500:700]]
        sec_dist_data = []
        secs = []
        for i in range(12530, 12630):
        # for i in rng:
            if self.secs_secs.get(i):
                print 'SECS DATA of %d:' % i, self.secs_secs[i]['dists']
                secs.append(i)
                sec_list.append(self.analyzer.get_a_sec_data_only(self.analyzer.game_data, i))
                sec_dist_data.append(self.secs_secs[i])

        ScenarioPlotter(sec_list, self.analyzer.team_ids, just_home, midway, sec_dist_data, secs,
                        self.def_cohesive_matrix, self.off_cohesive_matrix, self.names)

    def dist_plotter(self):
        DistTimePlotter({key: self.secs_secs[key]['dists'] for key in self.secs_secs})