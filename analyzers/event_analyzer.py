import pickle
import matplotlib.pyplot as plt
import pprint
import numpy as np
import traceback
import mysql.connector
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

    with open('../pickles/teams_events_real_spread.pkl', 'rb+') as f:
        data = pickle.load(f)


def interpolate(data, length):
    maxx = data[-1]
    minn = data[0]
    return [minn + i*((maxx - minn)/(length-1)) for i in range(length)]


def calculate_average_and_std(data):
    avg_data = []
    for i in range(20):
        avg_data.append(np.average([x[i] for x in data]))

    std = np.std(data, 0)
    std_up = [sum(x) for x in zip(avg_data, std)]
    std_bot = [x - y for x, y in zip(avg_data, std)]

    return avg_data, std_up, std_bot, np.average(std)


def scale_to_twenty_sec(flow):
    new_flow = [0]*20

    if len(flow) > 20:
        factor = (len(flow) - 1)/19.0
        i_range = 20
    else:
        factor = 19.0/(len(flow) - 1)
        i_range = len(flow)

    for i in range(i_range):
        ceil_index = int(ceil(factor*i))
        if len(flow) > 20:
            try:
                new_flow[i] = flow[ceil_index]
            except:
                print i, ceil_index, len(flow), factor, factor*i, int(ceil(factor*i)), int(factor*i)
        else:
            new_flow[ceil_index] = flow[i]

            if i != i_range - 1:
                cnt = 1
                for j in range(ceil_index+1, int(ceil(factor*(i+1)))):
                    eps = (flow[i+1] - flow[i])/(int(ceil(factor*(i+1))) - ceil_index)
                    new_flow[j] = flow[i] + (eps*cnt)
                    cnt += 1

    return new_flow


def ceil(x):
    if x - int(x) > 0.00001:
        return int(x) + 1
    return int(x)


def extract_png_graphs():
    team_averages = {}
    for team in data:
        print '--------------------------------------------------------------------------'
        print 'TEAM', team
        print '--------------------------------------------------------------------------'
        f, (axs1, axs2) = plt.subplots(2, 4)
        try:
            if not 10 in team_averages.keys():
                team_averages[10] = []
            successful_cross = map(lambda x: scale_to_twenty_sec(x), [row['flow'] for row in data[team].get(10, [])])
            successful_cross = filter(lambda x: not np.isnan(x[0]), successful_cross)
            for each in successful_cross:
                axs1[0].plot(each, '--')

            avg, std_up, std_bot, avg_std = calculate_average_and_std(successful_cross)
            team_averages[10].append((team, avg, avg_std))
            if team in top_five_ids:
                top_five[team] = {}
                top_five[team][10] = [avg, std_up, std_bot]
            if team in top_low_ids:
                top_low[team] = {}
                top_low[team][10] = [avg, std_up, std_bot]
            axs1[0].plot(avg, 'k')
            axs1[0].plot(std_up, 'k')
            axs1[0].plot(std_bot, 'k')
            axs1[0].set_title('Basarili Orta')
        except Exception, e:
            traceback.print_exc()
        try:
            if not 70 in team_averages.keys():
                team_averages[70] = []
            successful_shot = map(lambda x: scale_to_twenty_sec(x), [row['flow'] for row in data[team].get(70, [])])
            successful_shot = filter(lambda x: not np.isnan(x[0]), successful_shot)
            for each in successful_shot:
                axs1[1].plot(each, '--')
            avg, std_up, std_bot, avg_std = calculate_average_and_std(successful_shot)
            team_averages[70].append((team, avg, avg_std))
            if team in top_five_ids:
                top_five[team][70] = [avg, std_up, std_bot]
            if team in top_low_ids:
                top_low[team][70] = [avg, std_up, std_bot]
            axs1[1].plot(avg, 'k')
            axs1[1].plot(std_up, 'k')
            axs1[1].plot(std_bot, 'k')
            axs1[1].set_title('Basarili Sut')
        except:
            pass
        try:
            if not 72 in team_averages.keys():
                team_averages[72] = []
            unsuccessful_shot = map(lambda x: scale_to_twenty_sec(x), [row['flow'] for row in data[team].get(72)])
            unsuccessful_shot = filter(lambda x: not np.isnan(x[0]), unsuccessful_shot)
            for each in unsuccessful_shot:
                axs1[2].plot(each, '--')
            avg, std_up, std_bot, avg_std = calculate_average_and_std(unsuccessful_shot)
            team_averages[72].append((team, avg, avg_std))
            if team in top_five_ids:
                top_five[team][72] = [avg, std_up, std_bot]
            if team in top_low_ids:
                top_low[team][72] = [avg, std_up, std_bot]
            axs1[2].plot(avg, 'k')
            axs1[2].plot(std_up, 'k')
            axs1[2].plot(std_bot, 'k')
            axs1[2].set_title('Basarisiz Sut')
        except:
            pass
        try:
            if not 12 in team_averages.keys():
                team_averages[12] = []
            null_event = map(lambda x: scale_to_twenty_sec(x), [row['flow'] for row in data[team].get(12)])
            null_event = filter(lambda x: not np.isnan(x[0]), null_event)
            for each in null_event:
                axs1[3].plot(each, '--')
            avg, std_up, std_bot, avg_std = calculate_average_and_std(null_event)
            team_averages[12].append((team, avg, avg_std))
            if team in top_five_ids:
                top_five[team][12] = [avg, std_up, std_bot]
            if team in top_low_ids:
                top_low[team][12] = [avg, std_up, std_bot]
            axs1[3].plot(avg, 'k')
            axs1[3].plot(std_up, 'k')
            axs1[3].plot(std_bot, 'k')
            axs1[3].set_title('Top Kaybi')
        except:
            pass
        try:
            if not -10 in team_averages.keys():
                team_averages[-10] = []
            opp_successful_cross = map(lambda x: scale_to_twenty_sec(x), [row['flow'] for row in data[team].get(-10, [])])
            opp_successful_cross = filter(lambda x: not np.isnan(x[0]), opp_successful_cross)
            for each in opp_successful_cross:
                axs2[0].plot(each, '--')

            avg, std_up, std_bot, avg_std = calculate_average_and_std(opp_successful_cross)
            team_averages[-10].append((team, avg, avg_std))
            if team in top_five_ids:
                top_five[team] = {}
                top_five[team][-10] = [avg, std_up, std_bot]
            if team in top_low_ids:
                top_low[team] = {}
                top_low[team][-10] = [avg, std_up, std_bot]
            axs2[0].plot(avg, 'k')
            axs2[0].plot(std_up, 'k')
            axs2[0].plot(std_bot, 'k')
            axs2[0].set_title('Rak. Basarili Orta')
        except Exception, e:
            traceback.print_exc()
        try:
            if not -70 in team_averages.keys():
                team_averages[-70] = []
            opp_successful_shot = map(lambda x: scale_to_twenty_sec(x), [row['flow'] for row in data[team].get(-70, [])])
            opp_successful_shot = filter(lambda x: not np.isnan(x[0]), opp_successful_shot)
            for each in opp_successful_shot:
                axs2[1].plot(each, '--')
            avg, std_up, std_bot, avg_std = calculate_average_and_std(opp_successful_shot)
            team_averages[-70].append((team, avg, avg_std))
            if team in top_five_ids:
                top_five[team][-70] = [avg, std_up, std_bot]
            if team in top_low_ids:
                top_low[team][-70] = [avg, std_up, std_bot]
            axs2[1].plot(avg, 'k')
            axs2[1].plot(std_up, 'k')
            axs2[1].plot(std_bot, 'k')
            axs2[1].set_title('Rak. Basarili Sut')
        except:
            pass
        try:
            if not -72 in team_averages.keys():
                team_averages[-72] = []
            opp_unsuccessful_shot = map(lambda x: scale_to_twenty_sec(x), [row['flow'] for row in data[team].get(-72)])
            opp_unsuccessful_shot = filter(lambda x: not np.isnan(x[0]), opp_unsuccessful_shot)
            for each in opp_unsuccessful_shot:
                axs2[2].plot(each, '--')
            avg, std_up, std_bot, avg_std = calculate_average_and_std(opp_unsuccessful_shot)
            team_averages[-72].append((team, avg, avg_std))
            if team in top_five_ids:
                top_five[team][-72] = [avg, std_up, std_bot]
            if team in top_low_ids:
                top_low[team][-72] = [avg, std_up, std_bot]
            axs2[2].plot(avg, 'k')
            axs2[2].plot(std_up, 'k')
            axs2[2].plot(std_bot, 'k')
            axs2[2].set_title('Rak. Basarisiz Sut')
        except:
            pass
        try:
            if not -12 in team_averages.keys():
                team_averages[-12] = []
            opp_null_event = map(lambda x: scale_to_twenty_sec(x), [row['flow'] for row in data[team].get(-12)])
            opp_null_event = filter(lambda x: not np.isnan(x[0]), opp_null_event)
            for each in opp_null_event:
                axs2[3].plot(each, '--')
            avg, std_up, std_bot, avg_std = calculate_average_and_std(opp_null_event)
            team_averages[-12].append((team, avg, avg_std))
            if team in top_five_ids:
                top_five[team][-12] = [avg, std_up, std_bot]
            if team in top_low_ids:
                top_low[team][-12] = [avg, std_up, std_bot]
            axs2[3].plot(avg, 'k')
            axs2[3].plot(std_up, 'k')
            axs2[3].plot(std_bot, 'k')
            axs2[3].set_title('Rak. Top Kaybi')
        except:
            pass

        plt.savefig('%s.png' % teams.get(team, 'Unknown team'))
        plt.close(f)

    with open('../pickles/team_event_averages.pkl', 'w') as f:
        pickle.dump(team_averages, f)


def extract_comparisons(clr, data, name):
    for event in [[10, 'Orta'], [70, 'Basarili Sut'], [72, 'Basarisiz Sut'], [12, 'Null'],
                  [-10, 'Rak. Orta'], [-70, 'Rak. Bas. Sut'], [-72, 'Rak. Basz. Sut'], [-12, 'Rak. Null']]:
        f, ax = plt.subplots(1, 1)
        for c, team in enumerate(data):
            try:
                ax.plot(range(1,21), data[team][event[0]][1], '%s--' % clr[c], label=teams.get(team))
                ax.plot(range(1,21), data[team][event[0]][2], '%s--' % clr[c])
                ax.fill_between(range(1,21),
                                data[team][event[0]][1],
                                data[team][event[0]][2],
                                facecolor=clr[c], alpha=0.2)
            except:
                print '#' * 49
                print '#' * 49
                pprint.pprint(data[team])
                print event, team
                print '#' * 49
                print '#' * 49

        ax.legend()
        plt.savefig('%s_%s.png' % (name, event[1]))
        plt.close(f)


conn = MySqlConnection()
conn.execute_query('SELECT id, name FROM td_team')
teams = {}
for each in conn.get_cursor():
    teams[each[0]] = each[1]

top_five_ids = [1, 2, 3, 82, 139]
top_five_clr = ['r', 'b', 'y', 'g', 'c']
top_five = {}

top_low_ids = [1, 2, 71, 20004]
top_low_clr = ['r', 'b', 'y', 'g']
top_low = {}

load_pickle()
extract_png_graphs()
# extract_comparisons(top_five_clr, top_five, 'top_five')
# extract_comparisons(top_low_clr, top_low, 'top_low')