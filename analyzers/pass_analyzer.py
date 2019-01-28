from plotters.matrix_plotter import MatrixPlotter
from commons import Commons
import pprint

class PassAnalyzer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.p2p_dicts = []
        self.pass_matrices = None

        self.create_p2p_dicts()
        self.generate_pass_matrices()

    def create_p2p_dicts(self):
        for team in self.analyzer.teams:
            self.p2p_dicts.append(team.generate_p2p_dict())

    def check_conditions(self, *args):
        return Commons.check_same_team(*args) and \
               Commons.check_different_player(*args) and \
               Commons.check_few_sec_pass(*args)

    def add_to_matrix(self, team_info, team_data, row1, row2):
        player1 = team_info.get_player_name_by_jersey_number(row1[-1])
        player2 = team_info.get_player_name_by_jersey_number(row2[-1])

        if player1 and player2:
            team_data[player1][player2] += 1
            team_data[player2][player1] += 1

    def generate_pass_matrices(self):
        for c, team in enumerate(self.p2p_dicts):
            for cnt, row in enumerate(self.analyzer.ball_data[:-1]):
                if self.check_conditions(row, self.analyzer.ball_data[cnt+1]):
                    self.add_to_matrix(self.analyzer.teams[c], team, row, self.analyzer.ball_data[cnt+1])

        self.analyzer.pass_matrices = [
            Commons.dict_to_matrix(self.p2p_dicts[0], self.analyzer.teams[0].get_player_names(), True),
            Commons.dict_to_matrix(self.p2p_dicts[1], self.analyzer.teams[1].get_player_names(), True)
        ]


