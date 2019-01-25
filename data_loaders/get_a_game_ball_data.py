from sqls import Sqls
from db_data_collector import DbDataCollector
from pickle_loader import PickleLoader


class BallData(DbDataCollector):
    def __init__(self, get_type='Single', game=None):
        DbDataCollector.__init__(self, get_type, game)

    def get_data(self, file_name):
        if self.get_type == 'Single':
            self.pkl_loader = PickleLoader(file_name)
            self.db_data = self.pkl_loader.return_data()
            if self.db_data:
                self.ball_data = self.db_data['ball_data']
                self.db_data = self.db_data['teams']
            else:
                self.db_data = []
                self.get_data_from_db()
                self.pkl_loader.dump_data({'teams': self.db_data,
                                           'ball_data': self.ball_data})
        else:
            self.pkl_loader = PickleLoader('game_ball_%s_%s.pkl' % (self.team_abbs[0], self.team_abbs[1]))
            self.db_data = self.pkl_loader.return_data()
            if self.db_data:
                self.ball_data = self.db_data['ball_data']
                self.db_data = self.db_data['teams']
            else:
                print 'NOT FOUND PICKLE %s' % self.pkl_loader.filename
                self.db_data = []
                self.get_data_from_db(self.team_names[0], self.team_names[1])
                self.pkl_loader.dump_data({'teams': self.db_data,
                                           'ball_data': self.ball_data})

    def get_db_data(self):
        query = Sqls.GET_BALL_DATA
        self.conn.execute_query(query, (self.db_data[0].name, self.db_data[1].name))

    def process_data(self):
        for cnt, data in enumerate(self.conn.get_cursor()):
            self.ball_data.append([data[1], data[2], data[7], data[8]])
