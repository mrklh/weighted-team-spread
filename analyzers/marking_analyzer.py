import pprint
from plotters.dist_plotter import DistPlotter
from plotters.matrix_plotter import MatrixPlotter

class MarkingAnalyzer:
    def __init__(self, game_data):
        self.team_matrix = None
        self.team_matrices = []
        self.game_data = game_data
        self.home_id = self.game_data[0].id
        self.away_id = self.game_data[1].id
        self.tim_dict = {}
        self.create_team_matrices()
        self.generate_tim_dict()
        self.generate_marking_dicts()

    def generate_tim_dict(self):
        for team in self.game_data:
            for player_name in team.players:
                player = team.players[player_name]
                for sec_data in player.statistics_as_np:
                    if not self.tim_dict.get(int(sec_data[0])):
                        self.tim_dict[int(sec_data[0])] = {'players': {}, 'hasball_team_id': sec_data[-2]}

                    self.tim_dict[int(sec_data[0])]['players'][player.name] = \
                        [float(sec_data[4]), float(sec_data[5])]

    def create_team_matrices(self):
        for team in self.game_data:
            self.team_matrices.append(team.generate_teammates_matrix())

    def range_calculator(self, tim, player1, player2):
        if self.tim_dict[tim]['players'].get(player1) is None or \
                self.tim_dict[tim]['players'].get(player2) is None:
            return False

        point1 = self.tim_dict[tim]['players'][player1]
        point2 = self.tim_dict[tim]['players'][player2]

        if not sum(point1) or not sum(point2):
            return False

        return self.is_in_range(point1, point2)

    @staticmethod
    def is_in_range(point1, point2):
        if ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5 < 5:
            return True
        return False

    def generate_marking_dicts(self):
        for tim in self.tim_dict:
            for c, team in enumerate(self.game_data):
                player_marking_rivals = {}
                for player in team.get_player_names():
                    player_marking_rivals[player] = []
                    for rival in self.game_data[c ^ 1].get_player_names():
                        if self.range_calculator(tim, player, rival):
                            player_marking_rivals[player].append(rival)

                # print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++'
                # print player_marking_rivals
                # if sum([len(player_marking_rivals[x]) for x in player_marking_rivals]) > 4:
                #     dp = DistPlotter([],
                #                      [self.game_data_collector.db_data[0].id, self.game_data_collector.db_data[1].id],
                #                      normal_dists=[norm_dist_home, norm_dist_away],
                #                      my_dists=[my_dist_home, my_dist_away])
                #     dp.plot_pitch()
                #     dp.put_players_on_pitch()
                # print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++'
                hasball = int(self.tim_dict[tim]['hasball_team_id'])
                if not hasball or hasball == self.home_id:
                    self.analyze_markings(0, player_marking_rivals)

                if not hasball or hasball == self.away_id:
                    self.analyze_markings(1, player_marking_rivals)
        x = MatrixPlotter.dict_to_matrix(self.team_matrices[0], self.game_data[0].get_player_names(),
                                         is_normalize=False)

    def analyze_markings(self, ind, marking_data):
        players = self.game_data[ind].get_player_names()

        for p1 in players:
            for p2 in players:
                if p1 == p2:
                    continue
                try:
                    self.team_matrices[ind][p1][p2] += len(set(marking_data.get(p1, [])).
                                                           intersection(marking_data.get(p2, [])))

                except:
                    print 'HATAAAAA'
