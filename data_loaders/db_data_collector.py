# -*- coding: utf-8 -*-

from mine_sql_connection import MySqlConnection
from team import Team
from sqls import Sqls


class DbDataCollector:
    def __init__(self, get_type='Single', game=None):
        home_name = self.get_team_abb(game['home']['name'])
        away_name = self.get_team_abb(game['away']['name'])
        team_abbs = (home_name, away_name)
        team_names = (game['home']['name'], game['away']['name'])

        self.db_data = []
        self.match_id = game['id']
        self.ball_data = []
        self.conn = MySqlConnection()
        self.pkl_loader = None
        self.get_type = get_type
        self.team_names = team_names
        self.team_abbs = team_abbs

    def get_data(self, file_name):
        pass

    def get_data_from_db(self, team1=None, team2=None):
        if team1 and team2:
            self.create_teams(team1, team2)
        else:
            self.get_team_names_from_user()

        self.get_team_data()
        self.get_db_data()
        self.process_data()
        self.generate_teams_np_data()
        self.close_connection()

    def get_team_names_from_user(self):
        first_team = raw_input("Please enter first team name:\n")
        second_team = raw_input("Please enter second team name:\n")
        self.create_teams(first_team, second_team)

    def create_teams(self, first_team, second_team):
        self.db_data.append(Team(first_team))
        self.db_data.append(Team(second_team))

    def get_team_by_id(self, id):
        if self.db_data[0].id == id:
            return self.db_data[0]
        return self.db_data[1]

    def get_team_data(self):
        query = Sqls.GET_TEAM_DATA
        for team in self.db_data:
            self.conn.execute_query(query % team.name)
            for team_data in self.conn.get_cursor():
                team.set_team_id(team_data[0])

    def get_db_data(self):
        pass

    def process_data(self):
        pass

    def close_connection(self):
        self.conn.close_connection()

    def generate_teams_np_data(self):
        self.db_data[0].generate_np_data()
        self.db_data[1].generate_np_data()

    def get_player_data(self, pid):
        query = Sqls.GET_PLYR_DATA % pid

        self.conn.execute_query(query)

    @staticmethod
    def get_team_abb(team_name):
        if team_name == u'Beşiktaş':
            return 'BJK'
        if team_name == u'Galatasaray':
            return 'GS'
        if team_name == u'Fenerbahçe':
            return 'FB'
        if team_name == u'Akhisar Bld.Spor':
            return 'AKH'
        if team_name == u'Alanyaspor':
            return 'ALN'
        if team_name == u'Antalyaspor':
            return 'ANT'
        if team_name == u'Bursaspor':
            return 'BUR'
        if team_name == u'Gençlerbirliği':
            return 'GNC'
        if team_name == u'Göztepe':
            return 'GOZ'
        if team_name == u'Medipol Başakşehir':
            return 'BSK'
        if team_name == u'KDÇ Karabükspor':
            return 'KBK'
        if team_name == u'Kasımpaşa':
            return 'KSM'
        if team_name == u'Kayserispor':
            return 'KYS'
        if team_name == u'Atiker Konyaspor':
            return 'KON'
        if team_name == u'Yeni Malatyaspor':
            return 'MLT'
        if team_name == u'Sivasspor':
            return 'SVS'
        if team_name == u'Trabzonspor':
            return 'TS'
        if team_name == u'Osmanlıspor':
            return 'OSM'