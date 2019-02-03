import pickle
import matplotlib.pyplot as plt
import pprint
import numpy as np
import traceback
import mysql.connector
import math
import sys

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

def scale_to_twenty_sec(flow):
    print '/' * 50
    print '/' * 50
    pprint.pprint(flow)
    print '==', len(flow), '=='
    print '/' * 50
    print '/' * 50
    new_flow = [0]*20

    if len(flow) > 20:
        factor = (len(flow) - 1)/19.0
        i_range = 20
    else:
        factor = 19.0/(len(flow) - 1)
        i_range = len(flow)

    for i in range(i_range):
        ceil_index = int(math.ceil(factor*i))
        if len(flow) > 20:
            print 'ceil_index', ceil_index, 'i', i, 'factor', factor
            print 'flow[ceil_index]', flow[ceil_index]
            new_flow[i] = flow[ceil_index]
        else:
            print 'ceil_index', ceil_index, 'i', i, 'factor', factor
            print 'flow[i]', flow[i]
            new_flow[ceil_index] = flow[i]

            if i != i_range - 1:
                cnt = 1
                for j in range(ceil_index+1, int(math.ceil(factor*(i+1)))):
                    eps = (flow[i+1] - flow[i])/(int(math.ceil(factor*(i+1))) - ceil_index)
                    new_flow[j] = new_flow[i] + (eps*cnt)
                    cnt += 1

    print '#' * 50
    print '#' * 50
    pprint.pprint(new_flow)
    print '#' * 50
    print '#' * 50
    sys.exit(0)

def extract_png_graphs():
    for team in data:
        f, (axs1, axs2) = plt.subplots(2, 2, sharey=True)
        try:
            successful_cross = [row['flow'] for row in data[team].get(10, [])]
            for each in successful_cross:
                axs1[0].plot(scale_to_twenty_sec(each))
                sys.exit(0)
            # ax1.plot(calculate_average(successful_cross), 'k')
            axs1[0].set_title('Basarili Orta')
        except Exception, e:
            traceback.print_exc()
        try:
            successful_shot = [row['flow'] for row in data[team].get(70, [])]
            for each in successful_shot:
                axs1[1].plot(scale_to_twenty_sec(each))
                sys.exit(0)
            # ax2.plot(calculate_average(successful_shot), 'k')
            axs1[1].set_title('Basarili Sut')
        except:
            pass
        try:
            unsuccessful_shot = [row['flow'] for row in data[team].get(72)]
            for each in unsuccessful_shot:
                axs2[0].plot(scale_to_twenty_sec(each))
            # ax3.plot(calculate_average(unsuccessful_shot), 'k')
            axs2[0].set_title('Basarisiz Sut')
        except:
            pass
        try:
            null_event = [row['flow'] for row in data[team].get(12)]
            for each in null_event:
                axs2[1].plot(scale_to_twenty_sec(each))
            # ax3.plot(calculate_average(unsuccessful_shot), 'k')
            axs2[1].set_title('Top Kaybi')
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
