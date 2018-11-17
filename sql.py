#!/usr/bin/env python 
# coding: utf-8
# @Time : 2018/11/8 16:49 
# @Author : yangpingyan@gmail.com

from sshtunnel import SSHTunnelForwarder
import pandas as pd
import os
import json
from sqlalchemy import create_engine
from log import log
from mltools import *
import numpy as np
import time


def sql_connect(sql_file, ssh_pkey=None):
    '''连接数据库'''
    with open(sql_file, encoding='utf-8') as f:
        sql_info = json.load(f)

    ssh_host = sql_info['ssh_host']
    ssh_user = sql_info['ssh_user']
    sql_address = sql_info['sql_address']
    sql_user = sql_info['sql_user']
    sql_password = sql_info['sql_password']
    if ssh_pkey == None:
        sql_engine = create_engine(
            'mysql+pymysql://{}:{}@{}:3306/mibao_rds'.format(sql_user, sql_password, sql_address))
        log.debug("Access MySQL directly")
    else:
        server = SSHTunnelForwarder((ssh_host, 22),  # ssh的配置
                                    ssh_username=ssh_user,
                                    ssh_pkey=ssh_pkey,
                                    remote_bind_address=(sql_address, 3306))
        server.start()
        sql_engine = create_engine(
            'mysql+pymysql://{}:{}@127.0.0.1:{}/mibao_rds'.format(sql_user, sql_password, server.local_bind_port))
        log.debug("Access MySQL with SSH tunnel forward")

    return sql_engine
    # pd.read_sql("SELECT * from `order` o where o.id = 88668", sql_engine)


# 初始化数据库连接，使用pymysql模块
sql_file = os.path.join(workdir, 'sql_mibao.json')
ssh_pkey = os.path.join(workdir, 'sql_pkey') if debug_mode else None
sql_engine = sql_connect(sql_file, ssh_pkey)




'''
def read_sql(sql):

    timestart = time.clock()
    df = pd.read_sql_query(sql, sql_engine)
    print(time.clock() - timestart)

    df = pd.read_sql_query(sql, sql_engine)

    return df
'''

'''
获取订单和用户的相关信息只能是用户付款前的数据，涉及到的数据如下：
表 order： ['id', 'create_time', 'merchant_id', 'user_id', 'state', 'cost', 'installment', 'pay_num',
            'added_service', 'bounds_example_id', 'bounds_example_no', 'goods_type', 'lease_term',
            'commented', 'accident_insurance', 'type', 'order_type', 'device_type', 'source', 'distance',
            'disposable_payment_discount', 'disposable_payment_enabled', 'lease_num', 'merchant_store_id',
            'deposit', 'hit_merchant_white_list', 'fingerprint', 'cancel_reason', 'delivery_way',
            'order_number', 'joke']
表 user: ['id', 'head_image_url', 'recommend_code', 'regist_channel_type', 'share_callback', 'tag', 'phone']
表 bargain_help: ['user_id']
表 face_id:  ['user_id', 'status']
表 face_id_liveness: ['order_id', 'status']
表 user_credit: ['user_id', 'cert_no', 'workplace', 'idcard_pros', 'occupational_identity_type',
                'company_phone', 'cert_no_expiry_date', 'cert_no_json', ]
表 user_device: ['user_id', 'device_type', 'regist_device_info', 'regist_useragent', 'ingress_type'],
表 order_express: ['order_id', 'zmxy_score', 'card_id', 'phone', 'company']
表 order_detail: ['order_id', 'order_detail']
表 order_goods: ['order_id', 'price', 'category', 'old_level']
表 order_phone_book: ['order_id', 'phone_book']
表 risk_order: ['order_id', 'type', 'result', 'detail_json']
表 tongdun: ['order_number', 'final_score', 'final_decision']
表 user_third_party_account: ['user_id']
表 user_zhima_cert: ['user_id', 'status']
表 jimi_order_check_result_list: ['order_id', 'check_remark']

'''

'''
def get_data_sql(order_id=88667):
    # 读取order表

    features = ['o.id as order_id', 'o.create_time', 'o.merchant_id', 'o.user_id', 'o.state', 'o.cost', 'o.installment',
                'o.pay_num',
                'o.added_service', 'o.bounds_example_id', 'o.bounds_example_no', 'o.goods_type', 'o.lease_term',
                'o.commented', 'o.accident_insurance', 'o.type', 'o.order_type', 'o.device_type', 'o.source',
                'o.distance',
                'o.disposable_payment_discount', 'o.disposable_payment_enabled', 'o.lease_num', 'o.merchant_store_id',
                'o.deposit', 'o.hit_merchant_white_list', 'o.fingerprint', 'o.cancel_reason', 'o.delivery_way',
                'o.order_number',
                'e.zmxy_score', 'e.card_id', 'e.phone', 'e.company',
                'd.order_detail',
                'g.price', 'g.category', 'g.old_level'
                ]
    sql = "SELECT {} FROM `order` o " \
          "left join order_express e on o.id = e.order_id " \
          "left join order_detail d on o.id = d.order_id " \
          "left join order_goods g on o.id = g.order_id " \
          "WHERE o.joke = 0  and o.id = {}".format(
        ",".join(features), order_id)

    order_df = read_sql(sql)
    print(order_df)

    if len(order_df) == 0:
        return order_df
    user_id = order_df.at[0, 'user_id']
    order_number = order_df.at[0, 'order_number']
    all_data_df = order_df.copy()

    features = ['u.id as user_id', 'u.head_image_url', 'u.recommend_code', 'u.regist_channel_type', 'u.share_callback', 'u.tag', 'u.phone as phone_user',
                'f.status as face_check', 'fl.status as face_live_check',
                'uc.cert_no', 'uc.workplace', 'uc.idcard_pros', 'uc.occupational_identity_type',
                'uc.company_phone', 'uc.cert_no_expiry_date', 'uc.cert_no_json',
                'ud.device_type as device_type_os', 'ud.regist_device_info', 'ud.regist_useragent', 'ud.ingress_type',
                ]
    sql = "SELECT {} FROM `user` u " \
          "left join face_id f on u.id = f.user_id " \
          "left join face_id_liveness fl on u.id = fl.user_id " \
          "left join user_credit uc on u.id = uc.user_id " \
          "left join user_device ud on u.id = ud.user_id " \
          "WHERE u.id = {}".format(",".join(features), user_id)

    user_df = read_sql(sql)
    print(user_df)

    all_data_df = pd.merge(all_data_df, user_df, on='user_id', how='left')

    return all_data_df

   
  

    # 读取并处理表 bargain_help
    # bargain_help_df = read_sql('bargain_help', ['user_id'], 'user_id', user_id)
    all_data_df['have_bargain_help'] = np.where(all_data_df['user_id'].isin(bargain_help_df['user_id'].values), 1, 0)






    # 读取并处理表 order_phone_book
    # order_phone_book_df = read_sql('order_phone_book', ['order_id', 'phone_book'], 'order_id', order_id)
    # all_data_df = pd.merge(all_data_df, order_phone_book_df, on='order_id', how='left')
    # def count_name_nums(data):
    #     name_list = []
    #     if isinstance(data, str):
    #         data_list = json.loads(data)
    #         for phone_book in data_list:
    #             if len(phone_book.get('name')) > 0 and phone_book.get('name').isdigit() is False:
    #                 name_list.append(phone_book.get('name'))
    #
    #     return len(set(name_list))
    #
    # df['phone_book'] = df['phone_book'].map(count_name_nums)
    # df['phone_book'].fillna(value=0, inplace=True)

    # 读取并处理表 risk_order
    risk_order_df = read_sql('risk_order', ['order_id', 'type', 'result', 'detail_json'], 'order_id', order_id
                             )
    risk_order_df['result'] = risk_order_df['result'].str.lower()
    for risk_type in ['tongdun', 'mibao', 'guanzhu', 'bai_qi_shi']:
        tmp_df = risk_order_df[risk_order_df['type'].str.match(risk_type)][['order_id', 'result', 'detail_json']]
        tmp_df.rename(
            columns={'result': risk_type + '_result', 'detail_json': risk_type + '_detail_json'},
            inplace=True)
        all_data_df = pd.merge(all_data_df, tmp_df, on='order_id', how='left')
    # 读取并处理表 tongdun
    # tongdun_df = read_sql('tongdun', ['order_number', 'final_score', 'final_decision'], 'order_number', order_number
    #                       )
    all_data_df = pd.merge(all_data_df, tongdun_df, on='order_number', how='left')

    # 读取并处理表 user_third_party_account
    # user_third_party_account_df = read_sql('user_third_party_account', ['user_id'], 'user_id', user_id)
    # counts_df = pd.DataFrame({'account_num': user_third_party_account_df['user_id'].value_counts()})
    # counts_df['user_id'] = counts_df.index
    # all_data_df = pd.merge(all_data_df, counts_df, on='user_id', how='left')

    # 读取并处理表 user_zhima_cert
    # df = read_sql('user_zhima_cert', ['user_id', 'status'], 'user_id', user_id)
    all_data_df['zhima_cert_result'] = np.where(all_data_df['user_id'].isin(df['user_id'].tolist()), 1, 0)

    # 读取并处理表 jimi_order_check_result_list
    # df = read_sql('jimi_order_check_result', ['order_id', 'check_remark'], 'order_id', order_id)
    all_data_df = pd.merge(all_data_df, df, on='order_id', how='left')

    # 特殊字符串的列预先处理下：
    features = ['installment', 'commented', 'disposable_payment_enabled', 'face_check']
    # df = all_data_df.copy()
    for feature in features:
        # print(all_data_df[feature].value_counts())
        all_data_df[feature] = all_data_df[feature].astype(str)
        all_data_df[feature].fillna('0', inplace=True)
        all_data_df[feature] = np.where(all_data_df[feature].str.contains('1'), 1, 0)
        # print(all_data_df[feature].value_counts())

    return all_data_df
'''

