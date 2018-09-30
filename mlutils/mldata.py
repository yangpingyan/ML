#!/usr/bin/env python
# coding: utf-8
# @Time : 2018/9/28 16:42
# @Author : yangpingyan@gmail.com

import pandas as pd
from sqlalchemy import create_engine

# 初始化数据库连接，使用pymysql模块
engine = create_engine('mysql+pymysql://root:qawsedrf@localhost:3306/mibao')
datasets_path = "D:/datasets_ml/mibao/"

def load_data_mibao():
    df = load_data_mibao_sql()
    return df


def load_data_mibao_sql():
    # 查询语句，选出employee表中的所有数据
    sql = '''
          SELECT o.id,o.`create_time`,o.`merchant_id`,o.`user_id`,
o.`state`,o.`cost`,o.`discount`,o.`installment`,o.`pay_num`,o.`added_service`,o.`first_pay`,
o.`channel`,o.`pay_type`,o.`bounds_example_id`,o.`bounds_example_no`,o.`goods_type`,o.`cash_pledge`,
o.`cancel_reason`, o.`lease_term`,o.`commented`,o.`accident_insurance`,o.`type`,o.`freeze_money`,
o.`sign_state`,o.`ip`,o.`releted`,o.`order_type`,o.`device_type`,
o.`source`,o.`distance`,o.`disposable_payment_discount`,
o.`disposable_payment_enabled`,o.`lease_num`,o.`merchant_store_id`,o.`deposit`,
o.`hit_merchant_white_list`,o.`fingerprint`,
o.`hit_goods_white_list`,o.`credit_check_result`,
u.id AS u_id,u.`head_image_url`,u.`phone`,u.`code`,
u.`channel` as u_channel,u.`regist_ip`, u.`regist_channel_type`,
u.`residue_drawing_times`, u.`share_callback`, u.`tag`    FROM `order` o
LEFT JOIN `user` u ON o.`user_id` = u.`id`;
          '''
    # read_sql_query的两个参数: sql语句， 数据库连接
    df = pd.read_sql_query(sql, engine)

    return df


def save_all_tables_mibao():
    sql = ''' SELECT table_name FROM information_schema.`TABLES` WHERE table_schema="mibao"; '''
    tables_df = pd.read_sql_query(sql, engine)
    tables = tables_df['table_name'].values
    for table in tables:
        print(table)
        sql = "SELECT * FROM `{}`;".format(table)
        df = pd.read_sql_query(sql, engine)
        df.to_csv("{}{}.csv".format(datasets_path, table), index=False)


# 找出有user_id的表，然后只保存该表中有产生订单的user_id的相关数据
def read_save_infos_only_ordered():
    sql = ''' SELECT table_name, column_name FROM information_schema.columns  WHERE table_schema='mibao' AND column_name='user_id'; '''
    tables_df = pd.read_sql_query(sql, engine)
    tables = tables_df['table_name'].values.tolist()
    tables.remove('user_bonus')
    df = pd.read_csv("{}order.csv".format(datasets_path))
    user_ids = df['user_id'].unique()
    df = pd.read_csv("{}user.csv".format(datasets_path))
    df = df[df['id'].isin(user_ids)]
    df.to_csv("D:/datasets_ml/{}.csv".format("user"), index=False)
    for table in tables:
        print(table)
        df = pd.read_csv("{}{}.csv".format(datasets_path, table))
        df = df[df['user_id'].isin(user_ids)]
        df.to_csv("D:/datasets_ml/{}.csv".format(table), index=False)
