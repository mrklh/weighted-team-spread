import numpy as np
import pprint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.spatial import ConvexHull

from matrix_plotter import MatrixPlotter


class ScenarioPlotter:
    def __init__(self, sec_data_list, team_ids, just_home, midway, sec_dist_data, secs, def_matrix, off_matrix, names):
        self.sec_data_list = sec_data_list
        self.sec_count = len(sec_data_list)
        self.team_ids = team_ids
        self.secs = secs
        self.def_matrix = def_matrix
        self.off_matrix = off_matrix
        self.names = names
        self.fig = None
        self.pitch_ax = None
        self.mat_strength_ax = None
        self.mat_cohesive_ax = None
        self.dist_ax_l = None
        self.dist_ax_r = None
        self.home_sec_data = {}
        self.away_sec_data = {}
        self.convex_areas = []

        self.just_home = just_home
        self.midway = midway

        self.min_l = 0

        self.sec_dist_data = [data['dists'] for data in sec_dist_data]
        self.p2p_data = [data['p2p'] for data in sec_dist_data]

        self.lines = {'home': [], 'away': []}
        self.players = {'home': [], 'away': []}
        self.stats = {}
        self.cs_im = None
        self.coh_im = None
        self.time_text = None

        self.frobenius_plot = None
        self.my_dist_plot = None
        self.norm_dist_plot_x = None
        self.norm_dist_plot_y = None
        self.convex_plot = None

        self.hasball_data = []

        self.initialize_variables()
        self.interpolate_data()
        self.initialize_convex_data()
        self.plot_pitch()
        self.initialize_animation_variables()
        self.put_players_on_pitch()

    def interpolate_data(self):
        for player in self.home_sec_data:
            normal_data = self.home_sec_data[player]
            interpolated_data = []
            for i in range(len(normal_data) - 1):
                interpolated_data.append(normal_data[i])
                base = normal_data[i]
                interval = [(a_i - b_i) / 10.0 for a_i, b_i in zip(normal_data[i + 1], normal_data[i])]
                for j in range(1, 10):
                    interpolated_data.append([a_i + (b_i*j) for a_i, b_i in zip(base, interval)])
            interpolated_data.append(normal_data[-1])

            self.home_sec_data[player] = interpolated_data

        for player in self.away_sec_data:
            normal_data = self.away_sec_data[player]
            interpolated_data = []
            for i in range(len(normal_data) - 1):
                interpolated_data.append(normal_data[i])
                for j in range(1, 10):
                    base = normal_data[i]
                    interval = [a_i - b_i for a_i, b_i in zip(normal_data[i+1], normal_data[i])]
                    interpolated_data.append([a_i + (b_i*j) for a_i, b_i in zip(base, interval)])
            interpolated_data.append(normal_data[-1])

            self.away_sec_data[player] = interpolated_data

        self.min_l = min(min([len(self.away_sec_data[key]) for key in self.away_sec_data]),
                         min([len(self.home_sec_data[key]) for key in self.home_sec_data]))

    def initialize_variables(self):
        for c, sec_data in enumerate(self.sec_data_list):
            for player in sec_data:
                if player[0] == self.team_ids[0]:
                    # if not c:
                    #     self.names.append(player[-1] or 'Unknown Player')
                    data = self.home_sec_data
                else:
                    data = self.away_sec_data

                if player[-1] not in data:
                    data[player[-1]] = []

                data[player[-1]].append((player[5], player[6]))

        self.hasball_data = [x[0][-2] for x in self.sec_data_list]

    def initialize_convex_data(self):
        for sec_data in self.sec_data_list:
            hull = ConvexHull(np.array([[x[5], x[6]] for x in filter(lambda x: x[0] == self.team_ids[0], sec_data)]))
            self.convex_areas.append(hull.area/6.0)

    def initialize_pitch(self):
        for key in self.home_sec_data:
            line, = self.pitch_ax.plot([], [], 'ro')
            self.lines['home'].append(line)

            name_text = self.pitch_ax.text(0.0, 0.0, key or 'Unknown', color='red')
            self.players['home'].append(name_text)

        home_stat = self.pitch_ax.text(4.0, 4.0, '0.0 - 0.0')
        self.stats['home'] = home_stat

        if not self.just_home:
            for key in self.away_sec_data:
                line, = self.pitch_ax.plot([], [], 'bo')
                self.lines['away'].append(line)

                name_text = self.pitch_ax.text(0.0, 0.0, key or 'Unknown', color='blue')
                self.players['away'].append(name_text)

            away_stat = self.pitch_ax.text(80.0, 4.0, '0.0 - 0.0')
            self.stats['away'] = away_stat

        self.time_text = self.pitch_ax.text(55, 66.0, '{0:02d}:{0:02d}'.format(0, 0))

    def initialize_strength(self):
        init = np.zeros((11, 11))
        init[0][0] = -20
        init[0][1] = 20

        self.cs_im = self.mat_strength_ax.imshow(init, cmap='RdBu_r')
        plt.colorbar(self.cs_im, ax=self.mat_strength_ax)
        self.mat_strength_ax.set_title("(Cohesive Strength x Euc. Dist.) Matrix")
        self.mat_strength_ax.set_xticks(np.arange(0, 11, 1))
        self.mat_strength_ax.set_yticks(np.arange(0, 11, 1))
        self.mat_strength_ax.set_xticklabels(self.names)
        self.mat_strength_ax.set_yticklabels(self.names)
        for tick in self.mat_strength_ax.get_xticklabels():
            tick.set_rotation(90)
        for tick in self.mat_strength_ax.get_yticklabels():
            tick.set_rotation(45)

    def initialize_cohesive(self):
        init = np.zeros((11, 11))
        init[0][0] = 1

        self.coh_im = self.mat_cohesive_ax.imshow(init, cmap='RdBu_r')
        plt.colorbar(self.coh_im, ax=self.mat_cohesive_ax)
        self.mat_cohesive_ax.set_title("(Cohesive Strength x Euc. Dist.) Matrix")
        self.mat_cohesive_ax.set_xticks(np.arange(0, 11, 1))
        self.mat_cohesive_ax.set_yticks(np.arange(0, 11, 1))
        self.mat_cohesive_ax.set_xticklabels(self.names)
        self.mat_cohesive_ax.set_yticklabels(self.names)
        for tick in self.mat_cohesive_ax.get_xticklabels():
            tick.set_rotation(90)
        for tick in self.mat_cohesive_ax.get_yticklabels():
            tick.set_rotation(45)

    def initialize_dist_plot(self):
        self.dist_ax_l.set_xlim(1, self.min_l/10)
        self.dist_ax_l.set_ylim(0, 55)

        self.dist_ax_r.set_xlim(1, self.min_l/10)
        self.dist_ax_r.set_ylim(22, 31)

        self.frobenius_plot, = self.dist_ax_r.plot([], [], 'y', label='Frob./2')
        self.my_dist_plot, = self.dist_ax_r.plot([], [], 'r', label='My D.')
        self.norm_dist_plot_x, = self.dist_ax_l.plot([], [], 'g--', label='X. D.')
        self.norm_dist_plot_y, = self.dist_ax_l.plot([], [], 'b--', label='Y. D.')
        self.convex_plot, = self.dist_ax_r.plot([], [], 'k', label='Area/6')
        plt.legend(handles=[self.frobenius_plot,
                            self.my_dist_plot,
                            self.norm_dist_plot_x,
                            self.norm_dist_plot_y,
                            self.convex_plot])

    def initialize_animation_variables(self):
        self.initialize_pitch()
        self.initialize_strength()
        self.initialize_cohesive()
        self.initialize_dist_plot()

    def set_fig_ax(self, fig, axs):
        self.fig = fig
        axs[0][1].twinx()
        self.pitch_ax = fig.axes[0]
        self.dist_ax_l = fig.axes[1]
        self.mat_strength_ax = fig.axes[2]
        self.mat_cohesive_ax = fig.axes[3]
        self.dist_ax_r = fig.axes[4]

    def plot_pitch(self):
        fig, axs = plt.subplots(2, 2)
        self.set_fig_ax(fig, axs)

        img = plt.imread("resources/football_pitch.png")

        sec_data = self.secs[0]
        min0 = (sec_data % 10000) / 100
        sec0 = sec_data % 100

        sec_data = self.secs[-1]
        minn = (sec_data % 10000) / 100
        secn = sec_data % 100

        self.pitch_ax.set_title("Game Scenario {0:02d}:{1:02d}-{2:02d}:{3:02d}".format(min0, sec0, minn, secn))
        self.pitch_ax.imshow(img, extent=[0, 115, 0, 72])
        self.pitch_ax.set_xticks(np.arange(-2.0, 108, 10))
        self.pitch_ax.set_yticks(np.arange(-2, 73, 5))
        self.fig.tight_layout()

    def get_list_of_animes(self):
        return self.lines['home'] + \
               self.lines['away'] + \
               self.players['home'] + \
               self.players['away'] + \
               [self.stats[key] for key in self.stats] + \
               [self.time_text] + \
               [self.cs_im, self.coh_im] + \
               [self.frobenius_plot, self.my_dist_plot, self.norm_dist_plot_x, self.norm_dist_plot_y, self.convex_plot]

    def init(self):
        for lin in self.lines['home']:
            lin.set_data([], [])
        for lin in self.lines['away']:
            lin.set_data([], [])

        for name in self.players['home']:
            name.set_position((0, 0))
        for name in self.players['away']:
            name.set_position((0, 0))

        self.frobenius_plot.set_data([], [])
        self.my_dist_plot.set_data([], [])
        self.norm_dist_plot_x.set_data([], [])
        self.norm_dist_plot_y.set_data([], [])
        self.convex_plot.set_data([], [])

        self.cs_im.set_array(np.zeros((11, 11)))
        self.coh_im.set_array(np.zeros((11, 11)))

        return self.get_list_of_animes()

    def animate_pitch(self, i):
        for c, lin in enumerate(self.lines['home']):
            player = self.home_sec_data[self.home_sec_data.keys()[c]]
            x = player[i][0]
            y = 73 - player[i][1]

            lin.set_data([x], [y])
            self.players['home'][c].set_position((x+1, y+1))

        self.stats['home'].set_text('My D:%.2f - Norm:%.2f' % (self.sec_dist_data[i/10][0], self.sec_dist_data[i/10][2]))

        if not self.just_home:
            for c, lin in enumerate(self.lines['away']):
                player = self.away_sec_data[self.away_sec_data.keys()[c]]
                x = player[i][0]
                y = 73 - player[i][1]

                lin.set_data([x], [y])
                self.players['away'][c].set_position((x+1, y+1))

            if not self.just_home:
                self.stats['away'].set_text('My D:%.2f - Norm:%.2f' % (self.sec_dist_data[i/10][1], self.sec_dist_data[i/10][3]))

        self.set_time_text(i/10)

    def animate_strength(self, i):
        # import sys
        # # Calculating dist matrix
        # dist_matrix = {}
        # for p1 in self.names:
        #     if not dist_matrix.get(p1):
        #         dist_matrix[p1] = {}
        #     for p2 in self.names:
        #         p1_pos = self.home_sec_data[p1][i]
        #         p2_pos = self.home_sec_data[p2][i]
        #         dist_matrix[p1][p2] = ((p1_pos[0] - p2_pos[0]) ** 2 + (p1_pos[1] - p2_pos[1]) ** 2) ** 0.5
        #
        # dist_m = MatrixPlotter.dict_to_matrix(dist_matrix, self.names, is_normalize=False)
        # dist_tot = np.sum(dist_m)
        # wght_m = MatrixPlotter.dict_to_matrix(self.p2p_data[i / 10], self.names, is_normalize=False)
        # wght_tot = np.sum(wght_m)
        # dist_m = np.divide(dist_m, dist_tot/wght_tot)
        # diff_m = np.subtract(dist_m, wght_m)
        #
        # diff_def_v = np.vectorize(lambda v: 1 - v)
        # oneminusw = diff_def_v(wght_m)
        #
        # def frob(x, y):
        #     summ = 0
        #     for c1, i in enumerate(x):
        #         for c2, j in enumerate(i):
        #             summ += j * y[c1][c2]
        #     return summ
        # frob_diff = frob(dist_m, self.off_matrix)
        #
        # print '#' * 49
        # print '#' * 49
        # pprint.pprint(frob_diff)
        # print '#' * 49
        # print '#' * 49
        # # sys.exit(0)
        #
        # self.cs_im.set_array(np.subtract(dist_m, wght_m))
        self.cs_im.set_array(MatrixPlotter.dict_to_matrix(self.p2p_data[i / 10], self.names, is_normalize=False))

    def animate_cohesive(self, i):
        if self.hasball_data[i/10] == self.team_ids[0]:
            self.mat_cohesive_ax.set_title('Cohesive (OFF)')
            self.coh_im.set_array(self.off_matrix)
        else:
            self.mat_cohesive_ax.set_title('Cohesive (DEF)')
            self.coh_im.set_array(self.def_matrix)

    def animate_dist_plot(self, i):
        self.frobenius_plot.set_xdata(range(1, i/10+1))
        self.frobenius_plot.set_ydata([x[0]/2.0 for x in self.sec_dist_data[0:i/10]])

        self.my_dist_plot.set_xdata(range(1, i/10+1))
        self.my_dist_plot.set_ydata([x[1] for x in self.sec_dist_data[0:i/10]])

        self.norm_dist_plot_x.set_xdata(range(1, i/10+1))
        self.norm_dist_plot_x.set_ydata([x[4] for x in self.sec_dist_data[0:i/10]])

        self.norm_dist_plot_y.set_xdata(range(1, i/10+1))
        self.norm_dist_plot_y.set_ydata([x[5] for x in self.sec_dist_data[0:i/10]])

        self.convex_plot.set_xdata(range(1, i/10+1))
        self.convex_plot.set_ydata(self.convex_areas[0:i/10])

    def animate(self, i):
        self.animate_pitch(i)
        self.animate_strength(i)
        self.animate_cohesive(i)
        self.animate_dist_plot(i)

        return self.get_list_of_animes()

    def set_time_text(self, i):
        sec_data = self.secs[i]
        minn = (sec_data % 10000)/100
        sec = sec_data % 100
        self.time_text.set_text('{0:02d}:{1:02d}'.format(minn, sec))

    def put_players_on_pitch(self):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

        ani = animation.FuncAnimation(self.fig, self.animate, self.min_l,
                                      interval=100, blit=True, init_func=self.init, repeat_delay=10000)

        # ani.save('pitch.mp4', writer=writer)
        plt.show()

    def get_sec_dist_data(self, ind):
        return "%.2f - %.2f - %.2f - %.2f" % tuple(self.sec_dist_data[ind])

    def show(self):
        plt.show()

    def matrix_print(self, matrix, names=None):
        # print '      ', ''.join(map(lambda x: '{:<6}'.format(x), names))
        for each in matrix:
            for each2 in each:
                print '{:<6}'.format('%.2f' % each2),
            print '\n'
