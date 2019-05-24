import pickle
import matplotlib.pyplot as plt
import numpy as np
import pprint
import commons
import traceback

data = None
np.set_printoptions(precision=2)

def load_pickle():
    global data

    with open('../pickles/team_event_averages.pkl', 'rb+') as f:
        data = pickle.load(f)


def compare_on_off_ball(on, off):
    '''
    :param on: on ball flow
    :param off: of ball flow
    :return: True if on ball is smaller
    '''
    # print np.average(on), '<', np.average(off), np.average(on) < np.average(off)
    return np.average(on) < np.average(off)


def compare_end_on_off_ball(on, off):
    '''
    :param on: on ball flow
    :param off: of ball flow
    :return: True if on ball is smaller
    '''
    # print on[-1], '<', off[-1], on[-1] < off[-1]
    return on[-1] < off[-1]


def show_averages():
    def show_league_averages():
        # data to plot
        n_groups = 4
        means_commit = [np.average(avg[1]) for avg in league_averages][:4]
        means_suffer = [np.average(avg[1]) for avg in league_averages][4:]
        err_commit = [np.average(std[1]) for std in league_std][:4]
        err_suffer = [np.average(std[1]) for std in league_std][4:]

        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.8

        rects1 = plt.bar(index, means_commit, bar_width,
                         alpha=1,
                         color='#065819',
                         label='Committed',
                         yerr=err_commit,
                         align='center',
                         ecolor='black',
                         capsize=15)

        rects2 = plt.bar(index + bar_width, means_suffer, bar_width,
                         alpha=opacity/1.4,
                         color='#f05121',
                         label='Suffered',
                         yerr=err_suffer,
                         align='center',
                         ecolor='black',
                         capsize=15)

        plt.xlabel('Event Types', size=15)
        plt.ylabel('Average Team Spread Values', size=15)
        # plt.title('Average Team Spread Values of Turkey Super League', size=15)
        plt.xticks(index + bar_width/2, ('Succ. Cross', 'Shot on T.', 'Shot off T.', 'Dispossession'))
        plt.legend(loc=4)

        plt.tight_layout()
        plt.savefig('league_averages.pdf')
    # print '#' * 49
    # print '#' * 49
    # pprint.pprint(teams)
    # print '#' * 49
    # print '#' * 49

    # on ball off ball comparison variables
    suc_shot = 0
    suc_shot_end = 0
    unsuc_shot = 0
    unsuc_shot_end = 0
    suc_cross = 0
    suc_cross_end = 0
    null = 0
    null_end = 0

    team_averages = {}
    league_averages = []
    league_std = []

    events = [[10, 'Cross'], [70, 'Shot on T.'], [72, 'Shot off T.'], [12, 'Ball Loss'],
              [-10, 'S. Cross'], [-70, 'S. Shot on T.'], [-72, 'S. Shot off T.'], [-12, 'S. Ball Loss']]
    for c, event in enumerate(events):
        # f, (axs1, axs2) = plt.subplots(2, 2)
        # axes = [axs1[0], axs1[1], axs2[0], axs2[1]]
        teams = [team[0] for team in data[event[0]]]
        team_averages[event[0]] = []
        team_averages[-event[0]] = []
        for i, team in enumerate(teams):
            if not np.isnan(data[event[0]][i][1][0]):
                if compare_on_off_ball(data[event[0]][i][1], data[-event[0]][i][1]):
                    if event[0] == 10:
                        suc_cross += 1
                    elif event[0] == 12:
                        null += 1
                    elif event[0] == 70:
                        suc_shot += 1
                    elif event[0] == 72:
                        unsuc_shot += 1
                # print '////////////////////'
                if compare_end_on_off_ball(data[event[0]][i][1], data[-event[0]][i][1]):
                    if event[0] == 10:
                        suc_cross_end += 1
                    elif event[0] == 12:
                        null_end += 1
                    elif event[0] == 70:
                        suc_shot_end += 1
                    elif event[0] == 72:
                        unsuc_shot_end += 1

                team_averages[event[0]].append((team, data[event[0]][i][1]))
                team_averages[-event[0]].append((team, data[-event[0]][i][1]))
                # l1, = axes[c].plot(data[event[0]][i][1], 'r', label='Committed')
                # l2, = axes[c].plot(data[-event[0]][i][1], 'b', label='Suffered')
                # plt.legend(handles=[l1, l2])
                # axes[c].set_title(str(team) + ' - ' + event[1])

        # plt.show()

    for event in events:
        teams_data = [team[1] for team in team_averages[event[0]]]
        # print '#' * 49
        # print '#' * 49
        # pprint.pprint(teams)
        # print '#' * 49
        # print '#' * 49
        league_averages.append((event[0], np.average(teams_data, 0)))
        league_std.append((event[0], np.std(teams_data, 0)))

    show_league_averages()
    print '#' * 49
    print '#' * 49
    for c, each in enumerate(league_averages):
        print each[0], np.average(each[1]), np.average(league_std[c][1])
    print '#' * 49
    print '#' * 49
    print '\n==========================================================================='
    print '==========================================================================='
    print "suc_cross", suc_cross
    print "suc_shot", suc_shot
    print "unsuc_shot", unsuc_shot
    print "null", null
    print '==========================================================================='
    print "suc_cross end", suc_cross_end
    print "suc_shot end", suc_shot_end
    print "unsuc_shot end", unsuc_shot_end
    print "null end", null_end
    print '==========================================================================='
    print '===========================================================================\n'
    # teams = [1, 2, 3, 82, 90, 101, 105, 129]
    events = [[70, 'Shot on Target', (80, 120)], [12, 'Dispossession', (70, 110)],
              [-70, 'Suffered Shot on Target', (70, 110)], [-12, 'Tackles', (80, 120)]]

    for i in range(4):
        teams = [team[0] for team in data[events[i][0]]]
        selected_teams = commons.Commons.event_teams[events[i][0]]
        teams_data = [team for team in team_averages[events[i][0]]]
        for j in range(len(selected_teams)):
            try:
                fig, axs = plt.subplots(1, 1)
                team = selected_teams[j]
                index = teams.index(team)

                std_up = [sum(x) for x in zip(league_averages[2*i+1][1], league_std[2*i+1][1])]
                std_down = [x - y for x, y in zip(league_averages[2*i+1][1], league_std[2*i+1][1])]

                axs.set_ylabel('Team Spread Value', size=15)
                axs.set_xlabel('Time (scaled to 20 unit)', size=15)
                axs.plot(teams_data[index][1], 'b-*',
                                       label='Team Avg.')
                axs.plot(league_averages[2*i+1][1], 'r--', label='League Avg.')
                axs.plot(std_up, 'r', label='League Avg. $\pm$ std.')
                axs.plot(std_down, 'r')
                axs.set_ylim(events[i][2][0], events[i][2][1])
                axs.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
                axs.fill_between(range(0, 20),
                                std_up,
                                std_down,
                                facecolor='r', alpha=0.2)
                axs.legend(loc=1)
                plt.savefig('%s%s.pdf' % (commons.Commons.teams[team], events[i][1].replace(' ', '_')), bbox_inches='tight')
            except:
                traceback.print_exc()


load_pickle()
show_averages()
