import pprint
from plotters.dist_plotter import DistPlotter
from plotters.matrix_plotter import MatrixPlotter
from commons import Commons

class MarkingAnalyzer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.p2p_dicts = []
        self.marking_matrices = []
        self.tim_dict = {}

        self.create_p2p_dicts()
        self.generate_tim_dict()
        self.generate_marking_dicts()

    def generate_tim_dict(self):
        for team in self.analyzer.teams:
            for player_name in team.players:
                player = team.players[player_name]
                for sec_data in player.statistics_as_np:
                    if not self.tim_dict.get(int(sec_data[0])):
                        self.tim_dict[int(sec_data[0])] = {'players': {}, 'hasball_team_id': sec_data[-2]}

                    self.tim_dict[int(sec_data[0])]['players'][player.name] = \
                        [float(sec_data[4]), float(sec_data[5])]

    def create_p2p_dicts(self):
        for team in self.analyzer.teams:
            self.p2p_dicts.append(team.generate_p2p_dict())

    def range_calculator(self, tim, player1, player2):
        if self.tim_dict[tim]['players'].get(player1) is None or \
                self.tim_dict[tim]['players'].get(player2) is None:
            return False

        point1 = self.tim_dict[tim]['players'][player1]
        point2 = self.tim_dict[tim]['players'][player2]

        if not sum(point1) or not sum(point2):
            return False

        return Commons.is_in_range_mark(point1, point2)

    def generate_marking_dicts(self):
        '''

        For each sec:

        {
            player_1  : [marking_list_1],
            player_2  : [marking_list_2],
            .
            .
            .
            player_11 : [marking_list_11]
        }

        generated.

        Then

        :return:
        '''

        for tim in self.tim_dict:
            for c, team in enumerate(self.analyzer.teams):
                player_marking_rivals = {}
                for player in team.get_player_names():
                    player_marking_rivals[player] = []
                    for rival in self.analyzer.teams[c ^ 1].get_player_names():
                        if self.range_calculator(tim, player, rival):
                            player_marking_rivals[player].append(rival)

                '''
                If nobody has ball or opposite team has ball calculate markings.
                '''
                hasball = int(self.tim_dict[tim]['hasball_team_id'])
                if not hasball or hasball == self.analyzer.teams[0].id:
                    self.analyze_markings(1, player_marking_rivals)

                if not hasball or hasball == self.analyzer.teams[1].id:
                    self.analyze_markings(0, player_marking_rivals)

        self.analyzer.marking_matrices = [
            Commons.dict_to_matrix(self.p2p_dicts[0], self.analyzer.teams[0].get_player_names(), True),
            Commons.dict_to_matrix(self.p2p_dicts[1], self.analyzer.teams[1].get_player_names(), True)
        ]

    def analyze_markings(self, ind, marking_data):
        players = self.analyzer.teams[ind].get_player_names()

        for p1 in players:
            for p2 in players:
                if p1 == p2:
                    continue
                try:
                    self.p2p_dicts[ind][p1][p2] += len(set(marking_data.get(p1, [])).
                                                           intersection(marking_data.get(p2, [])))

                except:
                    print 'HATAAAAA'
