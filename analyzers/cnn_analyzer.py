# -*- coding: utf-8 -*-

import pickle
import pprint
import numpy as np
from math import isnan
import sys
import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras.utils import np_utils
from keras import optimizers
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import traceback
from keras.utils import plot_model

np.set_printoptions(precision=1)


class KernelViewer(keras.callbacks.Callback):
    def __init__(self, w1, w2):
        super(KernelViewer, self).__init__()
        self.w1 = w1
        self.w2 = w2

        self.fig, self.ax = plt.subplots()
        self.im = plt.imshow(self.kernel_to_img(), interpolation='none')

    def kernel_to_img(self):
        return [[x[0][0][0][0] for x in self.w1[0]], [x[0][0][0][0] for x in self.w2[0]]]

    def init(self):
        self.im.set_data(self.kernel_to_img())
        return [self.im]

    def animate(self, i):
        self.im.set_array(self.kernel_to_img())
        return [self.im]

    def on_train_begin(self, logs=None):
        self.init()
        ani = FuncAnimation(self.fig, self.animate, self.min_l,
                                      interval=100, blit=True, init_func=self.init, repeat_delay=10000)

    def on_epoch_end(self, epoch, logs=None):
        self.animate()


class CNN:
    def __init__(self, input1, input2, labels):
        self.input1 = input1
        self.input2 = input2
        self.labels = labels
        self.model = None

        self.conv1 = None
        self.conv2 = None

    def create_model(self):
        branch1 = Sequential()
        self.conv1 = Conv3D(1, (20, 1, 1), use_bias=False,
                            activation='relu',
                            kernel_initializer='glorot_uniform', input_shape=(20, 11, 11, 1))
        keras.layers.normalization.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)
        branch1.add(self.conv1)

        branch2 = Sequential()
        self.conv2 = Conv3D(1, (20, 1, 1), use_bias=False,
                            activation='relu',
                            kernel_initializer='glorot_uniform', input_shape=(20, 11, 11, 1))
        keras.layers.normalization.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)
        branch2.add(self.conv2)

        merged_out = Add()([branch1.output, branch2.output])
        merged_out = Flatten()(merged_out)
        merged_out = Activation('relu')(merged_out)
        keras.layers.normalization.BatchNormalization(epsilon=1e-06, mode=0, momentum=0.9, weights=None)
        merged_out = Dense(121, activation='softmax')(merged_out)
        # merged_out = Activation('relu')(merged_out)
        merged_out = Dense(2)(merged_out)

        self.model = Model([branch1.input, branch2.input], merged_out)

        self.model.summary()

        sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd,
                         loss='mean_squared_error',
                         metrics=['accuracy'])
        plot_model(self.model, to_file='model.png', show_shapes=True)

        # self.KernelViewer = KernelViewer(self.conv1.get_weights(), self.conv2.get_weights())

    def fit_and_print_kernels(self):
        # self.labels = np_utils.to_categorical(self.labels, 2)
        w1 = self.conv1.get_weights()
        w2 = self.conv2.get_weights()
        w11 = [x[0][0][0][0] for x in w1[0]]
        w21 = [x[0][0][0][0] for x in w2[0]]

        self.model.fit([self.input1, self.input2], self.labels,
                       epochs=4)

        w1 = self.conv1.get_weights()
        w2 = self.conv2.get_weights()
        w12 = [x[0][0][0][0] for x in w1[0]]
        w22 = [x[0][0][0][0] for x in w2[0]]

        fig, axs = plt.subplots(1, 1)
        axs.set_yticklabels(['', 'Passes', 'Closeness'])
        im3 = axs.imshow([[100*(x-y) for x, y in zip(w11, w12)], [100*(x-y) for x, y in zip(w21, w22)]], vmin=-1, vmax=1)
        plt.colorbar(im3)
        plt.savefig('kernels.pdf', bbox_inches='tight')

class CNN_Analyzer:
    def __init__(self):
        self.size_of_bulk = 0
        self.bulk = {}
        self.data1 = []
        self.data2 = []
        self.labels = []

    def get_input_and_labels(self):
        try:
            with open('../pickles/cnn_data.pkl', "rb") as f:
                cnn_data = pickle.load(f)
                self.data1 = cnn_data['data1']
                self.data2 = cnn_data['data2']
                print 'Normalization start'
                # self.normalize_input()
                print 'Normalization done.'
                self.labels = np_utils.to_categorical(cnn_data['labels'])
        except:
            traceback.print_exc()
            # self.get_bag_of_events()
            # self.prepare_input_and_labels()

    def get_bag_of_events(self):
        with open('../pickles/bag_of_events.pkl', 'rb') as f:
            self.bulk = pickle.load(f)

        self.size_of_bulk = len(self.bulk[10]) + len(self.bulk[70]) + len(self.bulk[72]) + len(self.bulk[12])

    def normalize(self, sec):
        def minmax(x):
            return (x - mn) / float(mx - mn)

        mx = np.max(sec)
        mn = np.min(sec)
        minmax_vec = np.vectorize(minmax)
        return minmax_vec(sec)

    def normalize_input(self):
        for i, scenario in enumerate(self.data1):
            for j, sec in enumerate(scenario):
                self.data1[i][j] = self.normalize(sec)
        for i, scenario in enumerate(self.data2):
            for j, sec in enumerate(scenario):
                self.data2[i][j] = self.normalize(sec)

    @staticmethod
    def progress(count, total, status=''):
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()

    @staticmethod
    def scale_vector_to_twenty_sec(flow):
        def ceil(x):
            if x - int(x) > 0.00001:
                return int(x) + 1
            return int(x)

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

    def scale_dist_flow_to_twenty_sec(self, dist_flow):
        new_dist_flow = np.zeros((20, 11, 11))
        for i in range(11):
            for j in range(11):
                flow = [x[i][j] for x in dist_flow]
                new_flow = self.scale_vector_to_twenty_sec(flow)
                for k in range(20):
                    try:
                        new_dist_flow[k][i][j] = new_flow[k]
                    except:
                        pass

        return new_dist_flow

    @staticmethod
    def is_passes_or_closeness_nan(passes, matrices):
        if isnan(passes[0][0]) or isnan(matrices[0][0]):
            return True
        return False

    def prepare_input_and_labels(self):
        cnt = 0
        for key in self.bulk:
            for event in self.bulk[key]:
                cnt += 1
                self.progress(cnt, self.size_of_bulk)
                if self.is_passes_or_closeness_nan(event['passes'], event['closeness']):
                    continue
                dist_flow = self.scale_dist_flow_to_twenty_sec(event['dist_flow'])

                self.data1.append(np.array([np.multiply(np.array(event['passes']), np.array(x)) for x in dist_flow]))
                self.data2.append(np.array([np.multiply(np.array(event['closeness']), np.array(x)) for x in dist_flow]))

                if key == 12:
                    self.labels.append(-1)
                else:
                    self.labels.append(1)

        self.data1 = np.array(self.data1).reshape(len(self.data1), 20, 11, 11, 1)
        self.data2 = np.array(self.data2).reshape(len(self.data2), 20, 11, 11, 1)

        with open('../pickles/cnn_data.pkl', 'wb') as f:
            pickle.dump({'data1': self.data1, 'data2': self.data2, 'labels': self.labels}, f)

    # TODO do cnn


cnna = CNN_Analyzer()
cnna.get_input_and_labels()

cnn = CNN(cnna.data1, cnna.data2, cnna.labels)
cnn.create_model()
cnn.fit_and_print_kernels()
