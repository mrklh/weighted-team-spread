from plotters.matrix_plotter import MatrixPlotter
import pprint
from commons import Commons

class ClosenessAnalyzer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.p2p_dicts = []
        self.closeness_matrices = None
        self.tim_dict = {}

        self.create_p2p_dicts()
        self.generate_tim_dict()
        self.generate_closeness_matrices()

    def generate_tim_dict(self):
        '''
        Generates
        
        {
            time : {
                hasball_team_id : id
                players : {
                    player1 : {
                        x, y
                    }            
                } 
            }
        }
        :return: 
        '''
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

    def range_calculator(self, range_type, tim, player1, player2):
        if self.tim_dict[tim]['players'].get(player1) is None or \
                self.tim_dict[tim]['players'].get(player2) is None:
            return False

        point1 = self.tim_dict[tim]['players'][player1]
        point2 = self.tim_dict[tim]['players'][player2]

        if not sum(point1) or not sum(point2):
            return False

        if range_type:
            x = Commons.is_in_range(point1, point2)
            return x
        return Commons.is_in_range_def(point1, point2)

    def generate_closeness_matrices(self):
        '''
        Generates p2p_dicts
        
        []
        
        :return: 
        '''
        for tim in self.tim_dict:
            for c, team in enumerate(self.p2p_dicts):
                range_type = True if self.tim_dict[tim]['hasball_team_id'] == self.analyzer.teams[c].id else False
                for player in team:
                    for friend in team[player]:
                        if self.range_calculator(range_type, tim, player, friend):
                            team[player][friend] += 1

        self.analyzer.closeness_matrices = [
            Commons.dict_to_matrix(self.p2p_dicts[0], self.analyzer.teams[0].get_player_names(), True),
            Commons.dict_to_matrix(self.p2p_dicts[1], self.analyzer.teams[1].get_player_names(), True)
        ]
