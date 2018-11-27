#!/usr/bin/env python
# coding: utf-8
# @Time : 2018/9/28 16:42
# @Author : yangpingyan@gmail.com

import csv
import json
import pandas as pd
import numpy as np
import os

from explore_data_utils import *
from mltools import *
from mldata import *
import operator
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# to make output display better
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 2000)

# read large csv file
csv.field_size_limit(100000000)

all_data_merged_df = get_order_data()
# In[]
df = all_data_merged_df.copy()

# 若state字段有新的状态产生， 抛出异常
state_values_newest = df['state'].unique().tolist()
assert (len(list(set(state_values_newest).difference(set(state_values)))) == 0)

# 丢弃不需要的数据
# 丢弃白名单用户
risk_white_list_df = read_mlfile('risk_white_list', ['user_id'])
user_ids = risk_white_list_df['user_id'].values
df = df[df['user_id'].isin(user_ids) != True]
# 丢弃joke为1的order
df = df[df['joke'] != 1]

# 标注人工审核结果于target字段
df['target'] = None
df.loc[df['state_cao'].isin(['manual_check_fail']), 'target'] = 0
df.loc[df['state_cao'].isin(['manual_check_success']), 'target'] = 1
df.loc[df['state'].isin(pass_state_values), 'target'] = 1
df.loc[df['state'].isin(failure_state_values), 'target'] = 0
df = df[df['target'].notnull()]
df['target'].value_counts()
df['state'].value_counts()

# 去除测试数据和内部员工数据
df = df[df['cancel_reason'].str.contains('测试') != True]
df = df[df['check_remark'].str.contains('测试') != True]
# 去除命中商户白名单的订单
df = df[df['hit_merchant_white_list'].str.contains('01') != True]

# 丢弃不需要的特征
df.drop(
    ['tongdun_detail_json', 'mibao_result', 'order_number', 'cancel_reason', 'hit_merchant_white_list', 'check_remark',
     'joke', 'mibao_remark', 'tongdun_remark', 'bai_qi_shi_remark', 'guanzhu_remark'],
    axis=1,
    inplace=True, errors='ignore')

# print(set(df.columns.tolist()) - set(df_sql.columns.tolist()))
# 保存数据
df.to_csv("mibaodata_merged.csv", index=False)
print("mibaodata_merged.csv saved with shape {}".format(df.shape))
# missing_values_table(df)


# 数据清洗
df = process_data_mibao(df)
df.to_csv(os.path.join(workdir, "mibaodata_ml.csv"), index=False)
print("mibaodata_ml.csv保存的数据量: {}".format(df.shape))
# In[]

exit('mergedata')

import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['font.sans-serif'] = ['Simhei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
