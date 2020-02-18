# -*- coding: utf-8 -*-
import traceback
import pprint

class Commons:
    teams = {
        1: 'Galatasaray',
        2: 'Fenerbahce',
        3: 'Besiktas',
        11: 'Akhisar Bld.Spor',
        13: 'Alanyaspor',
        21: 'Antalyaspor',
        42: 'Bursaspor',
        46: 'Caykur Rizespor',
        67: 'Gazisehir Gaziantep',
        71: 'Genclerbirligi',
        74: 'Goztepe',
        82: 'Medipol Basaksehir',
        86: 'KDC Karabukspor',
        90: 'Kasimpasa',
        93: 'Kayserispor',
        101: 'Atiker Konyaspor',
        105: 'Yeni Malatyaspor',
        129: 'Sivasspor',
        139: 'Trabzonspor',
        20004: 'Osmanlispor'
    }

    event_teams = {
        70: [1, 3, 13, 20004],
        12: [2, 74, 42, 86],
        -70: [3, 90, 21, 86],
        -12: [3, 74, 42, 86]
    }

    bjk_bsk = [
        'Fabricio',
        'Adriano',
        'Pepe',
        'Dusko',
        'Caner',
        'Atiba',
        'Tolgay',
        'Ricardo',
        'Anderson Souza ',
        'Jeremain',
        'Cenk'
    ]

    bjk_kon = [
        'Fabricio',
        'Adriano',
        'Pepe',
        'Gary',
        'Caner',
        'Atiba',
        'Tolgay',
        'Ricardo',
        'Anderson Souza ',
        'Ryan',
        'Cenk'
    ]

    bjk_kon_num = [
        1, 3, 5, 6, 88, 13, 18, 7, 94, 8, 23
    ]

    bjk_kon_jersey = {
        'Fabricio': 1,
        'Adriano': 3,
        'Pepe': 5,
        'Gary': 6,
        'Caner': 88,
        'Atiba': 13,
        'Tolgay': 18,
        'Ricardo': 7,
        'Anderson Souza ': 94,
        'Ryan': 8,
        'Cenk': 23
    }

    @staticmethod
    def check_same_team(row1, row2):
        return row1[-3] == row2[-3]

    @staticmethod
    def check_different_player(row1, row2):
        return row1[-1] != row2[-1]

    @staticmethod
    def check_few_sec_pass(row1, row2):
        return row1[0] - row2[0] < 3

    @staticmethod
    def is_in_range(point1, point2):
        if ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5 < 5:
            return True
        return False

    @staticmethod
    def is_in_range_def(point1, point2):
        if ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5 < 4:
            return True
        return False

    @staticmethod
    def is_in_range_mark(point1, point2):
        if ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5 < 2:
            return True
        return False

    @staticmethod
    def dict_to_matrix(data_as_dict, keys, is_normalize=True):
        import numpy as np

        def normalize():
            def minmax(x):
                if mx-mn == 0:
                    return 0
                return (x - mn) / float(mx - mn)

            mx = np.max(team_matrix)
            mn = np.min(team_matrix)
            minmax_vec = np.vectorize(minmax)
            return minmax_vec(team_matrix)

        team_matrix = np.zeros((11, 11))

        for cnt1, player1 in enumerate(keys):
            for cnt2, player2 in enumerate(keys):
                # If it is himself put 0 into that cell
                if player1 == player2:
                    team_matrix[cnt1, cnt2] = 0
                else:
                    try:
                        team_matrix[cnt1][cnt2] = data_as_dict[player1][player2]
                    except:
                        print '#' * 49
                        print '#' * 49
                        pprint.pprint(keys)
                        print '#' * 49
                        pprint.pprint(player1)
                        print '#' * 49
                        pprint.pprint(player2)
                        print '#' * 49
                        pprint.pprint(data_as_dict)
                        print '#' * 49
                        pprint.pprint(team_matrix)
                        print '#' * 49
                        print '#' * 49
        if is_normalize:
            team_matrix = normalize()
        return team_matrix

    @staticmethod
    def get_team_abb(team_name):
        if team_name == u'Beşiktaş':
            return 'BJK'
        if team_name == u'Galatasaray':
            return 'GS'
        if team_name == u'Fenerbahçe':
            return 'FB'
        if team_name == u'Akhisar Bld.Spor':
            return 'AKH'
        if team_name == u'Alanyaspor':
            return 'ALN'
        if team_name == u'Antalyaspor':
            return 'ANT'
        if team_name == u'Bursaspor':
            return 'BUR'
        if team_name == u'Gençlerbirliği':
            return 'GNC'
        if team_name == u'Göztepe':
            return 'GOZ'
        if team_name == u'Medipol Başakşehir':
            return 'BSK'
        if team_name == u'KDÇ Karabükspor':
            return 'KBK'
        if team_name == u'Kasımpaşa':
            return 'KSM'
        if team_name == u'Kayserispor':
            return 'KYS'
        if team_name == u'Atiker Konyaspor':
            return 'KON'
        if team_name == u'Yeni Malatyaspor':
            return 'MLT'
        if team_name == u'Sivasspor':
            return 'SVS'
        if team_name == u'Trabzonspor':
            return 'TS'
        if team_name == u'Osmanlıspor':
            return 'OSM'

    @staticmethod
    def decrease_time(time, minutes_base=100):
        half = time / 10000
        minute = (time - half * 10000) / minutes_base
        sec = time % 100

        if sec == 0:
            sec = 59
            if minute == 0:
                # Flow is cut
                return -1
            else:
                minute -= 1
        else:
            sec -= 1

        return half * 10000 + minute * minutes_base + sec

    @staticmethod
    def increase_time(time, minutes_base=100):
        half = time / 10000
        minute = (time - half * 10000) / minutes_base
        sec = time % 100

        if sec == 59:
            sec = 0
            minute += 1
        else:
            sec += 1

        return half * 10000 + minute * minutes_base + sec

