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
    engine = create_engine('mysql+pymysql://root:qawsedrf@localhost:3306/mibao')
    sql = ''' SELECT table_name FROM information_schema.`TABLES` WHERE table_schema="mibao"; '''
    tables_df = pd.read_sql_query(sql, engine)
    for table in tables_df['table_name'].values:
        print(table)
        sql = "SELECT * FROM `{}`;".format(table)
        df = pd.read_sql_query(sql, engine)
        df.to_csv("D:/datasets_ml/mibao/{}.csv".format(table), index=False)

