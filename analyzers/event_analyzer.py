import pickle
import matplotlib.pyplot as plt
import pprint
import numpy as np
import traceback
import mysql.connector


class MySqlConnection:
    def __init__(self):
        with open('../data_loaders/.db.config') as f:
            config = f.readlines()

        self.cnx = mysql.connector.connect(user=config[0].split('=')[1][:-1],
                                           password=config[1].split('=')[1][:-1],
                                           host=config[2].split('=')[1][:-1],
                                           database=config[3].split('=')[1][:-1],
                                           port=int(config[4].split('=')[1]))

        self.cursor = self.cnx.cursor()

    def execute_query(self, query, params=None):
        self.cursor.execute(query, params)

    def get_cursor(self):
        return self.cursor

    def close_connection(self):
        self.cursor.close()
        self.cnx.close()

data = None


def load_pickle():
    global data

    with open('../pickles/teams_events.pkl', 'rb+') as f:
        data = pickle.load(f)


def interpolate(data, length):
    maxx = data[-1]
    minn = data[0]
    return [minn + i*((maxx - minn)/(length-1)) for i in range(length)]


def calculate_average(data):
    max_len = max([len(row) for row in data])
    new_data = []
    for each in data:
        if len(each) < data:
            new_data.append(interpolate(each, max_len))

    avg_data = []
    for i in range(max_len):
        avg_data.append(np.average([x[i] for x in new_data]))

    return avg_data

def extract_png_graphs():
    for team in data:
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=True)
        try:
            successful_cross = [row['flow'] for row in data[team].get(10, [])]
            for each in successful_cross:
                ax1.plot(each)
            # ax1.plot(calculate_average(successful_cross), 'k')
            ax1.set_title('Basarili Orta')
        except Exception, e:
            traceback.print_exc()
        try:
            successful_shot = [row['flow'] for row in data[team].get(70, [])]
            for each in successful_shot:
                ax2.plot(each)
            # ax2.plot(calculate_average(successful_shot), 'k')
            ax2.set_title('Basarili Sut')
        except:
            pass
        try:
            unsuccessful_shot = [row['flow'] for row in data[team].get(72)]
            for each in unsuccessful_shot:
                ax3.plot(each)
            # ax3.plot(calculate_average(unsuccessful_shot), 'k')
            ax3.set_title('Basarisiz Sut')
        except:
            pass

        plt.savefig('%s.png' % teams.get(team, 'Unknown team'))
        plt.close(f)


conn = MySqlConnection()
conn.execute_query('SELECT id, name FROM td_team')
teams = {}
for each in conn.get_cursor():
    teams[each[0]] = each[1]

load_pickle()
extract_png_graphs()
