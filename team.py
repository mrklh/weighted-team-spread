import numpy as np

from player import Player


class Team:
    def __init__(self, name):
        self.name = name
        self.id = None
        self.players = {}
        self.player_names = []
        self.player_jersey_numbers = {}
        self.teammates_matrix = {}

    def set_team_id(self, id):
        self.id = id

    def add_player(self, name, jersey_number=None, formation=None):
        self.player_names.append(name)
        self.players[name] = Player(self.id, name)
        if jersey_number:
            self.player_jersey_numbers[jersey_number] = name

    def get_player(self, name):
        return self.players[name]

    def check_player_existance(self, name):
        return self.players.get(name) is not None

    def generate_np_data(self):
        for player in self.players:
            self.players[player].generate_np_data()

    def get_means(self):
        return np.array([self.players[player].mean for player in self.player_names])

    def get_player_names(self):
        return self.player_names

    def set_player_names(self, player_names):
        self.player_names = player_names

    def get_jersey_numbers(self):
        return self.player_jersey_numbers

    def set_jersey_numbers(self, jersey_numbers):
        self.player_jersey_numbers = jersey_numbers

    def generate_p2p_dict(self):
        self.teammates_matrix = {name1: {
            name2: 0 for name2 in self.player_names
        } for name1 in self.player_names}
        return self.teammates_matrix

    def get_player_name_by_jersey_number(self, num):
        return self.player_jersey_numbers.get(num)
