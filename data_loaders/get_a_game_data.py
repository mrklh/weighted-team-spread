# -*- coding: utf-8 -*-

import pprint

from db_data_collector import DbDataCollector
from sqls import Sqls
from pickle_loader import PickleLoader


class GameData(DbDataCollector):
    def __init__(self, get_type='Single', game=None):
        DbDataCollector.__init__(self, get_type, game)
        self.game_data = []
        self.events = []

    def get_data(self, file_name):
        if self.get_type == 'Single':
            self.pkl_loader = PickleLoader(file_name)
            self.db_data = self.pkl_loader.return_data()
            if not self.db_data:
                self.db_data = []
                self.get_data_from_db()
                self.get_events()
                self.pkl_loader.dump_data(self.db_data)
        else:
            self.pkl_loader = PickleLoader('game_%s_%s.pkl' % (self.team_abbs[0], self.team_abbs[1]))
            self.db_data = self.pkl_loader.return_data()
            if self.db_data:
                self.game_data = self.db_data['game_data']
                self.db_data = self.db_data['db_data']
                # self.events = self.db_data['events']
            else:
                print 'NOT FOUND PICKLE %s', self.pkl_loader.filename
                self.db_data = []
                self.get_data_from_db(self.team_names[0], self.team_names[1])
                self.pkl_loader.dump_data({'db_data': self.db_data,
                                           'game_data': self.game_data,})
                                           # 'events': self.events})

    def get_db_data(self):
        query = Sqls.GET_GAME_DATA
        self.conn.execute_query(query, (self.db_data[0].name, self.db_data[1].name))

    def process_data(self):
        for cnt, data in enumerate(self.conn.get_cursor()):
            self.game_data.append(data)
            player_name = data[-1] or "Unknown Player"
            team = self.get_team_by_id(data[0])
            row_data = [data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]]

            if not team.check_player_existance(player_name):
                team.add_player(player_name, data[7])

            team.get_player(player_name).append_to_statistics(row_data)

    def generate_teams_np_data(self):
        self.db_data[0].generate_np_data()
        self.db_data[1].generate_np_data()

    def get_events(self):
        query = Sqls.GET_EVENT_DATA
        self.conn.execute_query(query, self.match_id)

        for event in self.conn.get_cursor():
            self.events.append(event)
