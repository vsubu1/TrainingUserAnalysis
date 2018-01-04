# SQLite_LoadData.py
# This module opens connection to SQLite DB and returns the results set. It has two methods.
# First one takes tbale name as parameter and returns the result set for the table.
# The Second one takes SQL string as parameter and returns the result set.
import pandas as pd
import csv, sqlite3
import logging

import sqlite3
import os.path


def loadData(tableName) :
    # get relative path
    cpath = os.getcwd()
    m = cpath.rfind('/')
    cpath = cpath[:m]
    db_path = cpath + "/db/training.db"
    print(db_path)
    #now connect
    with sqlite3.connect(db_path) as conn:
        #conn = sqlite3.connect('pluralsight.db')
        c = conn.cursor()

        sql = "SELECT * FROM " + tableName
        df = pd.read_sql_query(sql , conn)
        print(df.head())
        return (df)
        c.close()
        conn.close()


def loadDatabySQL(sql) :
    # get relative path
    cpath = os.getcwd()
    m = cpath.rfind('/')
    cpath = cpath[:m]
    db_path = cpath + "/db/training.db"
    print("Sql = " + sql)
    print("Db path= " + db_path)
    # now connect
    with sqlite3.connect(db_path) as conn:
        #conn = sqlite3.connect('pluralsight.db')
        c = conn.cursor()
        df = pd.read_sql_query(sql , conn)
        print(df.head())
        return (df)
        c.close()
        conn.close()
