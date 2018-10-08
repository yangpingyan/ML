#!/usr/bin/env python 
# coding: utf-8
# @Time : 2018/9/30 13:39 
# @Author : yangpingyan@gmail.com

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
df.to_csv(datasets_path + "mibao.csv", index=False)
print("mibao.csv saved")

'''
features_jimi_order_check_result = ['check_result', 'check_remark', 'order_id']
jimi_order_check_result_df = pd.read_csv(datasets_path + "jimi_order_check_result.csv")
jimi_order_check_result_df = jimi_order_check_result_df[features_jimi_order_check_result]

df = pd.merge(order_df, jimi_order_check_result_df, on='order_id', how='left')

jimi_order_check_result_df['check_remark'].value_counts()

features_merchant = ['check_result', 'check_remark', 'order_id']
merchant_df = pd.read_csv(datasets_path + "merchant.csv")
merchant_df = merchant_df[features_merchant]
merchant_df['temp_risk_level'].value_counts()


df['check_result'].fillna(value='INIT')
feature_analyse(df, 'check_result')
df[df['check_result'].isnull()].sort_values(by='state').shape
df['credit_check_result'].value_counts()
df.shape
df['state'].value_counts()
missing_values_table(df)
df['state'].unique()
'''
