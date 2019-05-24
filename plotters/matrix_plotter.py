# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from commons import Commons
import pprint

class MatrixPlotter:
    def __init__(self):
        self.keys = None
        self.closeness_matrix = None
        self.pass_matrix = None
        self.marking_matrix = None
        self.scatter = None
        self.matrix = None

    def set_keys(self, keys):
        self.keys = keys

    def set_closeness_matrix(self, matrix):
        self.closeness_matrix = matrix

    def set_pass_matrix(self, matrix):
        self.pass_matrix = matrix

    def set_marking_matrix(self, matrix):
        self.marking_matrix = matrix

    def set_scatter(self, scatter):
        self.scatter = scatter

    def shorten_names(self):
        names = []
        for name in self.keys:
            try:
                name = name.strip().split(' ')[-1]
            except:
                name = unicode.strip(name)
            names.append(name)

        return names

    def plot(self):
        print '#' * 49
        print '#' * 49
        pprint.pprint('PLOTTING')
        print '#' * 49
        print '#' * 49
        names = self.shorten_names()
        grid = gridspec.GridSpec(1, 1)
        fig = plt.figure()

        ax1 = fig.add_subplot(grid[0, 0])
        plt.xticks(size=15)
        plt.yticks(size=15)
        matrix = Commons.dict_to_matrix(self.closeness_matrix, self.keys)
        self.matrix = matrix
        im = ax1.imshow(matrix, cmap='Greens')
        plt.colorbar(im)
        ax1.set_title("Closeness Matrix", size=15)
        ax1.set_xticks(np.arange(0, 11, 1))
        ax1.set_yticks(np.arange(0, 11, 1))
        ax1.set_xticklabels(names)
        ax1.set_yticklabels(names)
        for tick in ax1.get_xticklabels():
            tick.set_rotation(-90)

        plt.savefig("close1.pdf", bbox_inches='tight')

        grid = gridspec.GridSpec(1, 1)
        fig = plt.figure()

        ax2 = fig.add_subplot(grid[0, 0])
        matrix = Commons.dict_to_matrix(self.pass_matrix, self.keys)
        im = ax2.imshow(matrix, cmap='Greens')
        plt.colorbar(im)
        ax2.set_title("Pass Counts Matrix", size=15)
        ax2.set_xticks(np.arange(0, 11, 1))
        ax2.set_yticks(np.arange(0, 11, 1))
        ax2.set_xticklabels(names)
        ax2.set_yticklabels(names)
        for tick in ax2.get_xticklabels():
            tick.set_rotation(-90)

        plt.savefig("pass1.pdf", bbox_inches='tight')
        grid = gridspec.GridSpec(1, 1)
        fig = plt.figure()

        ax2 = fig.add_subplot(grid[0, 0])
        matrix = Commons.dict_to_matrix(self.marking_matrix, self.keys)
        im = ax2.imshow(matrix, cmap='Greens')
        plt.colorbar(im)
        ax2.set_title("Marking Matrix", size=15)
        ax2.set_xticks(np.arange(0, 11, 1))
        ax2.set_yticks(np.arange(0, 11, 1))
        ax2.set_xticklabels(names)
        ax2.set_yticklabels(names)
        for tick in ax2.get_xticklabels():
            tick.set_rotation(-90)

        plt.savefig("mark1.pdf", bbox_inches='tight')

    def plot_weights(self):
        # team_eleven = {
        #     'Fabricio': [10,34],
        #     self.keys[4]: [30, 22],
        #     'Adriano': [30, 84],
        #     'Atiba': [50, 43],
        #     'Domagoj': [30, 40],
        #     'Dusko': [30, 60],
        #     'Gary': [50, 66],
        #     'Ricardo': [75, 22],
        #     'Ryan': [75, 84],
        #     'Tolgay': [75, 53],
        #     'Vagner': [90, 35]
        # }
        #
        # print self.keys
        # print self.matrix
        #
        # for name1 in team_eleven:
        #     for name2 in team_eleven:
        #         plt.plot(team_eleven[name1],
        #                  team_eleven[name2],
        #                  'r--',
        #                  linewidth=self.matrix[self.keys.count(name1), self.keys.count(name2)])
        #
        # plt.show()
        self.plot_scatter()

    def plot_scatter(self, importance_list):
        fig, ax = plt.subplots(1, 1)

        self.scatter.set_fig_ax(fig, ax)
        self.scatter.display_scatter(importance_list)

        plt.show()
