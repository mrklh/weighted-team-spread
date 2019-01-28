import pprint

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
        time = rtime = event_time
        init_hasball_team = 0

        while True:
            hasball_team = self.find_who_has_ball(time)
            if not init_hasball_team and hasball_team:
                init_hasball_team = hasball_team

            # Veri bozulduysa return
            if time not in sec_keys:
                return rtime


            if hasball_team != init_hasball_team:
                return rtime

            rtime = time
            time = Commons.decrease_time(time)

            if time < 0:
                return rtime

    def find_who_has_ball(self, time):
        try:
            data = self.analyzer.ball_data[self.sec_keys_of_ball_data.index(time)]
            return data[2]
        except:
            return -1

    def trace_game_events(self):
        '''
        Traces all events and finds if it has continuous series of seconds of 10
        '''
        sec_keys = sorted(self.secs_secs.keys())

        for event in self.analyzer.events:
            self.find_beginning_of_attacking_transaction(event[0])
            continuous = True
            for i in range(8):
                if not (event[0] - i) in sec_keys:
                    continuous = False
                    continue
            if continuous:
                index = 1 if event[1] == self.analyzer.teams[0].id else 3
                base = self.secs_secs[event[0] - 7]['dists'][index]
                event_flow = [self.secs_secs[x]['dists'][index] - base for x in range(event[0] - 7, event[0])]

                if not self.analyzer.events_by_type.get(event[-1]):
                    self.analyzer.events_by_type[event[-1]] = []
                self.analyzer.events_by_type[event[-1]].append({'event': event, 'flow': event_flow})

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