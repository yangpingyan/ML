#!/usr/bin/env python
# coding: utf-8
# @Time : 2018/9/28 16:42
# @Author : yangpingyan@gmail.com

import pandas as pd
from sqlalchemy import create_engine

def load_data_mibao():
    df = load_data_mibao_sql()
    return df

def load_data_mibao_sql():
    # 初始化数据库连接，使用pymysql模块
    engine = create_engine('mysql+pymysql://root:qawsedrf@localhost:3306/mibao')

    # 查询语句，选出employee表中的所有数据
    sql = '''
          select * from `admin`;
          '''

    # read_sql_query的两个参数: sql语句， 数据库连接
    df = pd.read_sql_query(sql, engine)

    return df

