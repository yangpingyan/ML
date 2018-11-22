# coding: utf-8

import csv
import json
import pandas as pd
import numpy as np
import os
# Suppress warnings
from mltools import *
from mldata import *
import operator
import time




# read large csv file
csv.field_size_limit(100000000)

starttime = time.clock()
all_data_df = get_order_data()
print(time.clock() - starttime)
all_data_df.shape

# 若state字段有新的状态产生， 抛出异常
state_values_newest = all_data_df['state'].unique().tolist()
assert (len(list(set(state_values_newest).difference(set(state_values)))) == 0)

# 丢弃不需要的数据
# 丢弃白名单用户
df = read_mlfile('risk_white_list', ['user_id'])
user_ids = df['user_id'].values
all_data_df = all_data_df[all_data_df['user_id'].isin(user_ids) != True]
# 丢弃joke非0的order
all_data_df = all_data_df[all_data_df['joke'] == 0]
# 丢弃未做人工审核的order
all_data_df = all_data_df[all_data_df['state'].isin(pending_state_values + ['system_credit_check_unpass_canceled']) != True]

# 根据state生成target，代表最终审核是否通过
all_data_df.insert(0, 'target', np.where(all_data_df['state'].isin(failure_state_values), 0, 1))

# 去除测试数据和内部员工数据
all_data_df = all_data_df[all_data_df['cancel_reason'].str.contains('测试') != True]
all_data_df = all_data_df[all_data_df['check_remark'].str.contains('测试') != True]
# 去除命中商户白名单的订单
all_data_df = all_data_df[all_data_df['hit_merchant_white_list'].str.contains('01') != True]

# 丢弃不需要的特征
all_data_df.drop(
    ['tongdun_detail_json', 'mibao_result', 'order_number', 'cancel_reason', 'hit_merchant_white_list', 'check_remark',
     'joke', 'mibao_remark', 'tongdun_remark', 'bai_qi_shi_remark', 'guanzhu_remark'],
    axis=1,
    inplace=True, errors='ignore')

# print(set(all_data_df.columns.tolist()) - set(all_data_df_sql.columns.tolist()))
# 保存数据
all_data_df.to_csv("mibaodata_merged.csv", index=False)
print("mibao.csv saved with shape {}".format(all_data_df.shape))
# missing_values_table(all_data_df)


# In[]
exit('mergedata')
