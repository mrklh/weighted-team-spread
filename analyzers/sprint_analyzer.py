import pickle
import os

import numpy as np

from data_loaders.mine_sql_connection import MySqlConnection
from data_loaders.sqls import Sqls

POSITIONS = [
    'GK',
    'FB',
    'LB',
    'RB',
    'DMC',
    'CM',
    'AMC',
    'LW',
    'RW',
    'FW',
    'U'
]

players = {}
stats = []
s_max = 0
s_min = 5
total1 = 0
total2 = 0

# get data
files = os.listdir("../pickles/")
for s_file in files:
    if s_file.endswith("_sprints.pkl"):
        with open("../pickles/%s" % s_file, "rb") as f:
            data = pickle.load(f)  # {match_id + "_" + team_id + "_" + jersey: [list_of_sprint_values]}

            # find max and min sprint in the match
            for player in data:
                for sprint in data[player]:
                    if sprint > s_max:
                        s_max = sprint

                    if sprint < s_min:
                        s_min = sprint

            # normalize values in match, feed players own sprint values list
            for player in data:
                data[player] = [(x-s_min)/(s_max - s_min) for x in data[player]]

                total1 += len(data[player])
                if not players.get(player[6:]):
                    players[player[6:]] = []

                players[player[6:]] += data[player]

# get player names
conn = MySqlConnection("../data_loaders/.db.config")
conn.execute_query(Sqls.GET_ROSTERS_OF_TEAMS)
rosters = conn.get_cursor().fetchall()

# get player positions
with open("../pickles/positions", "r") as f:
    player_names = f.readlines()

new_rosters = {}
for i, x in enumerate(rosters):
    new_rosters[str(x[3]) + "_" + str(x[2])] = (x[0] + " " + x[1], player_names[i][:-1])

# normalize data
# for player in players:
#     for sprint in players[player]:
#         if sprint > s_max:
#             s_max = sprint
#
#         if sprint < s_min:
#             s_min = sprint
#
# for player in players:
#     players[player] = [(x-s_min)/(s_max - s_min) for x in players[player]]

# find averages and standard deviations
for player in players:
    np_list = np.array(players[player])
    stats.append((player, np.average(np_list), np.std(np_list), len(players[player]), new_rosters[player]))

stats = sorted(stats, key=lambda pl: pl[1])

positional_stats = [[] for i in range(len(POSITIONS))]
for stat in stats:
    positional_stats[POSITIONS.index(stat[-1][1])].append(stat[1])

positional_stats = [sum(x)/len(x) for x in positional_stats]