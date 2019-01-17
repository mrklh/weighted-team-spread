class PassAnalyzer:
    def __init__(self, game_data, team):
        self.team = team
        self.game_data = game_data
        self.teammate_matrix = self.team.generate_teammates_matrix()

    def analyze_data(self):
        for cnt, row in enumerate(self.game_data[:-1]):
            if self.check_conditions(row, self.game_data[cnt+1]):
                self.add_to_matrix(row, self.game_data[cnt+1])

    def check_conditions(self, *args):
        return self.check_same_team(*args) and \
               self.check_different_player(*args) and \
               self.check_few_sec_pass(*args)

    def add_to_matrix(self, row1, row2):
        player1 = self.team.get_player_name_by_jersey_number(row1[-1])
        player2 = self.team.get_player_name_by_jersey_number(row2[-1])

        if player1 and player2:
            self.teammate_matrix[player1][player2] += 1
            self.teammate_matrix[player2][player1] += 1

    @staticmethod
    def check_same_team(row1, row2):
        return row1[-2] == row2[-2]

    @staticmethod
    def check_different_player(row1, row2):
        return row1[-1] != row2[-1]

    @staticmethod
    def check_few_sec_pass(row1, row2):
        return row1[0] - row2[0] < 3
