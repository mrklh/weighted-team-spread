import pprint


class ClosenessAnalyzer:
    def __init__(self, game_data):
        self.team_matrix = None
        self.team_matrices = []
        self.game_data = game_data
        self.tim_dict = {}
        self.create_team_matrices()
        self.generate_tim_dict()
        self.generate_closeness_dicts()

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

    def range_calculator(self, typ, tim, player1, player2):
        if self.tim_dict[tim]['players'].get(player1) is None or \
                self.tim_dict[tim]['players'].get(player2) is None:
            return False

        point1 = self.tim_dict[tim]['players'][player1]
        point2 = self.tim_dict[tim]['players'][player2]

        if not sum(point1) or not sum(point2):
            return False

        if typ:
            x = self.is_in_range_off(point1, point2)
            return x
        return self.is_in_range_def(point1, point2)

    @staticmethod
    def is_in_range_off(point1, point2):
        if ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5 < 5:
            return True
        return False

    @staticmethod
    def is_in_range_def(point1, point2):
        if ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5 < 1:
            return True
        return False

    def generate_closeness_dicts(self):
        for tim in self.tim_dict:
            for c, team in enumerate(self.team_matrices):
                range_type = True if self.tim_dict[tim]['hasball_team_id'] == self.game_data[c].id else False
                for player in team:
                    for friend in team[player]:
                        if self.range_calculator(range_type, tim, player, friend):
                            team[player][friend] += 1
