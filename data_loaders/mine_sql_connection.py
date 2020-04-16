# -*- coding: utf-8 -*-

import mysql.connector


class MySqlConnection:
    def __init__(self, path=""):
        if not path:
            path = 'data_loaders/.db.config'
        with open(path) as f:
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
