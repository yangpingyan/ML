#!/usr/bin/env python
# coding: utf-8
# @Time : 2018/9/30 13:39
# @Author : yangpingyan@gmail.com

import numpy as np
import pandas as pd
import warnings
import os
import operator
from mlutils import *

warnings.filterwarnings('ignore')
# to make output display better
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 2000)
PROJECT_ID = 'mibao'

# ## 获取数据
if os.getcwd().find(PROJECT_ID) == -1:
    os.chdir(PROJECT_ID)
datasets_path = os.getcwd() + '\\datasets\\'

features_order = ['id', 'create_time', 'merchant_id', 'user_id', 'state', 'cost', 'installment', 'pay_num',
                  'added_service', 'bounds_example_id', 'bounds_example_no', 'goods_type', 'lease_term',
                  'commented', 'accident_insurance', 'type', 'ip', 'order_type', 'device_type', 'source', 'distance',
                  'disposable_payment_discount', 'disposable_payment_enabled', 'lease_num', 'merchant_store_id',
                  'deposit', 'hit_merchant_white_list', 'fingerprint', ]
order_df = pd.read_csv(datasets_path + "order.csv", encoding='utf-8', engine='python')
order_df = order_df[features_order]
order_df.rename(columns={'id': 'order_id'}, inplace=True)

df = order_df
# 根据state生成TARGET，代表最终审核是否通过
state_values = ['pending_receive_goods', 'running', 'user_canceled', 'pending_pay',
                'artificial_credit_check_unpass_canceled', 'pending_artificial_credit_check', 'lease_finished',
                'return_overdue', 'order_payment_overtime_canceled', 'pending_send_goods',
                'merchant_not_yet_send_canceled', 'running_overdue', 'buyout_finished', 'pending_user_compensate',
                'repairing', 'express_rejection_canceled', 'pending_return', 'returning', 'return_goods',
                'pending_relet_check', 'returned_received', 'relet_finished', 'merchant_relet_check_unpass_canceled',
                'system_credit_check_unpass_canceled', 'pending_jimi_credit_check', 'pending_relet_start',
                'pending_refund_deposit', 'merchant_credit_check_unpass_canceled']
failure_state_values = ['user_canceled', 'artificial_credit_check_unpass_canceled', 'return_overdue', 'running_overdue',
                        'merchant_relet_check_unpass_canceled', 'system_credit_check_unpass_canceled',
                        'merchant_credit_check_unpass_canceled']
pending_state_values = ['pending_artificial_credit_check', 'pending_relet_check', 'pending_jimi_credit_check',
                        'pending_relet_start']
state_values_newest = df['state'].unique().tolist()
assert (operator.eq(state_values_newest, state_values))


def state_mapping(value):
    if value in failure_state_values:
        return 0
    elif value in pending_state_values:
        return 2
    else:
        return 1

    return 0


df.insert(0, 'TARGET', df['state'].map(state_mapping))
df.drop(['state'], axis=1, inplace=True)

# read user data
user_df = pd.read_csv(datasets_path + "user.csv")
user_df['head_image_url'].fillna(value=0, inplace=True)
user_df['have_head_image'] = user_df['head_image_url'].map(lambda x: 0 if x == ("headImg/20171126/ll15fap1o16y9zfr0ggl3g8xptgo80k9jbnp591d.png") or x == 0 else 1)
user_df['recommend_code'].fillna(value=0, inplace=True)
user_df['recommend_code'] = user_df['recommend_code'].map(lambda x: 0 if x == 0 else 1)
user_df['share_callback'] = user_df['recommend_code'].map(lambda x: 1 if x > 0 else 0)
features_user = ['id', 'have_head_image', 'recommend_code', 'regist_channel_type', 'share_callback', 'tag']
user_df = user_df[features_user]
df = pd.merge(order_df, user_df, left_on='user_id', right_on='id', how='left')
df.drop(['id'], axis=1, inplace=True)

# read bargain_help data
bargain_help_df = pd.read_csv(datasets_path + "bargain_help.csv")
df['have_bargain_help'] = np.where(df['user_id'].isin(bargain_help_df['user_id'].values), 1, 0)

# read faceid data




df.to_csv(datasets_path + "mibao.csv", index=False)
print("mibao.csv saved")



'''


df.columns.values
feature = 'tag'
df[feature].value_counts()
df[feature].fillna(value='INIT', inplace=True)
feature_analyse(df, feature)
df[df[feature].isnull()].sort_values(by='state').shape

missing_values_table(df)
df[feature].unique()
'''
