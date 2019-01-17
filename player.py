import numpy as np


class Player:
    def __init__(self, team, name):
        self.team = team
        self.name = name
        self.statistics = []
        self.statistics_as_np = None
        self.position_data = None
        self.mean = None
        self.sec_list = {}

    def __repr__(self):
        return self.name.encode('utf-8')

    def append_to_statistics(self, data):
        self.statistics.append(data)

    def generate_np_data(self):
        self.statistics_as_np = np.array(self.statistics)
        self.position_data = np.array(self.statistics_as_np[:, [4, 5]],
                                      dtype='float')
        self.generate_avg_of_np_data()
        self.generate_sec_list()

    def generate_sec_list(self):
        for sec in self.statistics:
            self.sec_list[sec[0]] = (sec[4], sec[5])

    def generate_avg_of_np_data(self):
        self.mean = np.mean(self.position_data, 0)
