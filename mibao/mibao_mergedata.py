#!/usr/bin/env python 
# coding: utf-8
# @Time : 2018/9/30 13:39 
# @Author : yangpingyan@gmail.com

import pandas as pd
import warnings
import os
from mlutils import *

warnings.filterwarnings('ignore')
# to make output display better
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 2000)
PROJECT_ID = 'mibao'

# ## 获取数据
if os.getcwd().find(PROJECT_ID) == -1:
    os.chdir(PROJECT_ID)
datasets_path = os.getcwd() + '\\datasets\\'


features_order = ['id', 'create_time', 'merchant_id', 'user_id', 'state', 'cost', 'discount', 'installment',
                  'pay_num', 'added_service', 'first_pay', 'channel', 'pay_type', 'bounds_example_id',
                  'bounds_example_no', 'goods_type', 'cash_pledge', 'cancel_reason', 'lease_term', 'commented',
                  'accident_insurance', 'type', 'freeze_money', 'sign_state', 'ip', 'releted', 'order_type',
                  'device_type', 'source', 'distance', 'disposable_payment_discount', 'disposable_payment_enabled',
                  'lease_num', 'merchant_store_id', 'deposit', 'hit_merchant_white_list', 'fingerprint',
                  'hit_goods_white_list', 'credit_check_result']
order_df = pd.read_csv(datasets_path+"order.csv", encoding='utf-8', engine='python')
order_df = order_df[features_order]
order_df.rename(columns={'id':'order_id'}, inplace=True)

features_jimi_order_check_result = ['check_result', 'check_remark', 'order_id']
jimi_order_check_result_df = pd.read_csv(datasets_path+"jimi_order_check_result.csv", encoding='utf-8', engine='python')
jimi_order_check_result_df = jimi_order_check_result_df[features_jimi_order_check_result]

df = pd.merge(order_df, jimi_order_check_result_df, on='order_id', how='left')
# df = df[df['check_result'].str.contains('SUCCESS|FAILURE', na=False)]

value_mapping = {"SUCCESS": 1, "FAILURE": 0}
df.insert(0, 'TARGET', df['check_result'].map(value_mapping))
df.drop(columns='check_result', inplace=True)
df.loc[df['state'].str.contains('overdue') == True, 'TARGET'] = 0



df.to_csv(datasets_path+"mibao.csv", index=False)
feature_analyse(df, 'state')
df[df['check_result'].isnull()].sort_values(by='state').shape