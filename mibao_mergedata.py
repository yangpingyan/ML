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
df.loc[df['state_cao'] == 'manual_check_success', 'target'] = 1
df.loc[df['state'].isin(pass_state_values), 'target'] = 1
df.loc[df['state'].isin(failure_state_values), 'target'] = 0
df = df[df['target'].notnull()]
df['target'].value_counts()

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

# 机器学习
from sklearn.model_selection import train_test_split
import lightgbm as lgb

print(list(set(df.columns.tolist()).difference(set(mibao_ml_features))))

x = df[mibao_ml_features]
y = df['target'].tolist()
# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

score_df = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'])

# Create a training and testing dataset
train_set = lgb.Dataset(data=x_train, label=y_train)
test_set = lgb.Dataset(data=x_test, label=y_test)
# Get default hyperparameters
lgb_clf = lgb.LGBMClassifier()
lgb_params = lgb_clf.get_params()
# Number of estimators will be found using early stopping
if 'n_estimators' in lgb_params.keys():
    del lgb_params['n_estimators']
    # Perform n_folds cross validation
# param['metric'] = ['auc', 'binary_logloss']


lgb_params_binary_logloss = lgb_params.copy()
ret = lgb.cv(lgb_params, train_set, num_boost_round=10000, nfold=5, early_stopping_rounds=100, metrics='binary_logloss')
lgb_params_binary_logloss['n_estimators'] = len(ret['binary_logloss-mean'])
# Train and make predicions with model
lgb_clf = lgb.LGBMClassifier(**lgb_params_binary_logloss)
lgb_clf.fit(x_train, y_train)
y_pred = lgb_clf.predict(x_test)
add_score(score_df, 'binary_logloss', y_test, y_pred)

lgb_params_auc = lgb_params.copy()
ret = lgb.cv(lgb_params, train_set, num_boost_round=10000, nfold=5, early_stopping_rounds=100, metrics='auc')
lgb_params_auc['n_estimators'] = len(ret['auc-mean'])
# Train and make predicions with model
lgb_clf = lgb.LGBMClassifier(**lgb_params_auc)
lgb_clf.fit(x_train, y_train)
y_pred = lgb_clf.predict(x_test)
add_score(score_df, 'auc', y_test, y_pred)
# save model
# pickle.dump(lgb_clf, open('mibao_ml.pkl', 'wb'))
with open('lgb_params.json', 'w') as f:
    json.dump(lgb_params_auc, f, indent=4)

feature_importances = lgb_clf.feature_importances_
importance_df = pd.DataFrame({'name': x_train.columns, 'importance': feature_importances})
importance_df.sort_values(by=['importance'], ascending=False, inplace=True)
print(importance_df)

y_pred = lgb_clf.predict(x)
add_score(score_df, 'auc_alldata', y_test=y, y_pred=y_pred)
print(score_df)

# In[]
'''
                accuracy  precision    recall        f1             confusion_matrix
binary_logloss  0.980428   0.935366  0.934227  0.934796      [[4593, 53], [54, 767]]
auc             0.980794   0.934466  0.937881  0.936170      [[4592, 54], [51, 770]]
auc_alldata     0.985202   0.952223  0.948611  0.950414  [[46108, 389], [420, 7753]]

                accuracy  precision    recall        f1             confusion_matrix
binary_logloss  0.984989   0.944169  0.932598  0.938348      [[5801, 45], [55, 761]]
auc             0.985440   0.945477  0.935049  0.940234      [[5802, 44], [53, 763]]
auc_alldata     0.987091   0.951699  0.942616  0.947135  [[58055, 391], [469, 7704]]

                accuracy  precision    recall        f1             confusion_matrix
binary_logloss  0.982783   0.932039  0.919760  0.925859      [[6253, 56], [67, 768]]
auc             0.982783   0.929952  0.922156  0.926037      [[6251, 58], [65, 770]]
auc_alldata     0.987640   0.951002  0.940414  0.945678  [[62870, 396], [487, 7686]]
'''
exit('mergedata')

import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['font.sans-serif'] = ['Simhei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
