#!/usr/bin/env python 
# coding: utf-8
# @Time : 2018/11/8 16:49 
# @Author : yangpingyan@gmail.com

# import MySQLdb
from sshtunnel import SSHTunnelForwarder
import pandas as pd
import os
# import mlutils
import json
from sqlalchemy import create_engine

'''
def get_workdir(projectid):
    cur_dir = os.getcwd()
    if cur_dir.find(projectid) == -1:
        cur_dir = os.path.join(cur_dir, projectid)
    return cur_dir


workdir = get_workdir('mibao')
sql_file = os.path.join(workdir, 'sql_mibao.json')
ssh_pkey = os.path.join(workdir, 'sql_pkey')
'''


def sql_connect(sql_file, ssh_pkey=None):
    '''连接数据库'''
    with open(sql_file, encoding='utf-8') as f:
        sql_info = json.load(f)

    ssh_host = sql_info['ssh_host']
    ssh_user = sql_info['ssh_user']
    sql_address = sql_info['sql_address']
    sql_user = sql_info['sql_user']
    sql_password = sql_info['sql_password']
    if ssh_pkey is None:
        sql_engine = create_engine(
            'mysql+pymysql://{}:{}@{}:3306/mibao_rds'.format(sql_user, sql_password, sql_address))
    else:
        server = SSHTunnelForwarder((ssh_host, 22),  # ssh的配置
                                    ssh_username=ssh_user,
                                    ssh_pkey=ssh_pkey,
                                    remote_bind_address=(sql_address, 3306))
        server.start()
        sql_engine = create_engine(
            'mysql+pymysql://{}:{}@127.0.0.1:{}/mibao_rds'.format(sql_user, sql_password, server.local_bind_port))

    return sql_engine
    # pd.read_sql("SELECT * from `order` o where o.id = 88668", sql_engine)
