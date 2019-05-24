from data_loaders.mine_sql_connection import MySqlConnection
from data_loaders.sqls import Sqls
from data_loaders.pickle_loader import PickleLoader


class Validator:
    def __init__(self):
        self.conn = MySqlConnection()
        self.games = None
        self.game_dict = {}
        self.pkl_loader = PickleLoader('games.pkl')

    def get_games_from_db(self):
        self.games = self.pkl_loader.return_data()

        if not self.games:
            print 'NOT FOUND PICKLE GAMES'
            self.games = []
            self.conn.execute_query(Sqls.GET_GAMES_DATA)

            for game in self.conn.get_cursor():
                self.games.append(game)

            # self.pkl_loader.dump_data(self.games)

    def return_games(self):
        self.get_game_info()
        return self.game_dict

    def get_game_info(self):
        for game in self.games:
            self.game_dict[game[0]] = {
                'home': {'name': game[1], 'id': game[2]},
                'away': {'name': game[3], 'id': game[4]},
                'score': str(game[-2]) + '-' + str(game[-1]),
                'id': game[0]
            }