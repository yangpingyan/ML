#!/usr/bin/env python 
# coding: utf-8
# @Time : 2018/11/12 10:10 
# @Author : yangpingyan@gmail.com

# coding:utf8
import os
import time
from argparse import ArgumentParser
from flask import Flask, jsonify
from flask import make_response
import pandas as pd
import json
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
from gevent.pywsgi import WSGIServer
from mltools import *
from mldata import *
import logging
from mibao_log import log
import random
from explore_data_utils import *
from sql import *

warnings.filterwarnings('ignore')

ml_start_order_id = 109957
pred_start_order_id = 109975
# 查验审核准确度
sql = '''
    SELECT o.id as 'order_id', o.`create_time`, o.state, r.`type`, r.`result`, r.`remark`, cao.state as 'state_cao', cao.`remark` as 'remark_cao', o.deposit FROM `order` o 
LEFT JOIN risk_order r ON r.`order_id` = o.id
LEFT JOIN credit_audit_order cao ON cao.`order_id` = o.id
WHERE o.id > {} 
ORDER BY o.state DESC;
    '''.format(pred_start_order_id)
# 108034
# 108425

df = read_sql_query(sql)
df['state_cao'].value_counts()
# 标注人工审核结果于target字段
df['target'] = None
df.loc[df['state_cao'] == 'manual_check_fail', 'target'] = 0
df.loc[df['state_cao'] == 'manual_check_success', 'target'] = 1
df.loc[df['state'].isin(pass_state_values), 'target'] = 1
df.loc[df['state'].isin(failure_state_values), 'target'] = 0
df['target'].value_counts()
df[df['state_cao'] == 'manual_check_fail']
df = df[df['order_id'].isin(df[df['remark'].isin(['需人审'])]['order_id'].tolist() )]
df = df[df['type'].isin(['data_works'])]
df.sort_values(by='target', inplace = True, ascending=False)

manual_check_df = df[df['state_cao'].isin(['manual_check_fail', 'manual_check_success'])]
manual_check_df['target'] = manual_check_df['target'].astype(int)
manual_check_df['result'] = manual_check_df['result'].astype(int)
manual_check_df['result_cmp'] = manual_check_df['target'] - manual_check_df['result']

score_df = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'])
add_score(score_df, 'manual_check', manual_check_df['target'].astype(int).tolist(), manual_check_df['result'].astype(int).tolist())

df['target'].fillna(0, inplace=True)
df['target'] = df['target'].astype(int)
df['result'] = df['result'].astype(int)
df['result_cmp'] = df['target'] - df['result']

add_score(score_df, 'all_check', df['target'].astype(int).tolist(), df['result'].astype(int).tolist())



# In[]
# 验证机器学习结果是否一致

# 获取训练数据
all_data_ml_df = pd.read_csv(os.path.join(workdir, "mibaodata_ml.csv"), encoding='utf-8', engine='python')
df = all_data_ml_df[all_data_ml_df['order_id'] <= ml_start_order_id]
print("数据量: {}".format(df.shape))
x = df[mibao_ml_features]
y = df['target'].tolist()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# 机器学习模型训练
with open(os.path.join(workdir, "lgb_params.json"), 'r') as f:
    lgb_params_auc = json.load(f)

lgb_clf = lgb.LGBMClassifier(**lgb_params_auc)
lgb_clf.fit(x_train, y_train)
y_pred = lgb_clf.predict(x_test)
add_score(score_df, 'pred_train', y_test, y_pred)

order_ids = manual_check_df['order_id'].tolist()
df = all_data_ml_df[all_data_ml_df['order_id'].isin(order_ids)]
x = df[mibao_ml_features].copy()
y = df['target'].tolist().copy()
y_pred = lgb_clf.predict(x)
add_score(score_df, 'pred_check', y, y_pred)
df['pred_result'] = y_pred
manual_check_df = pd.merge(manual_check_df, df[['order_id', 'pred_result']], on='order_id', how='left')

manual_check_df[manual_check_df['result'] != manual_check_df['pred_result']]



# In[]
# 检查数据清洗是否正确

# 获取训练数据
all_data_ml_df = pd.read_csv(os.path.join(workdir, "mibaodata_ml.csv"), encoding='utf-8', engine='python')
print("数据量: {}".format(all_data_ml_df.shape))

# 模型测试：逐一读取数据库文件， 检验数据处理结果与机器学习处理结果是否一直
# order_ids = random.sample(all_data_ml_df['order_id'].tolist(), 1000)
# order_ids = [49769, 54841, 91984, 63778, 40925, 64166, 26342, 76788, 95580, 56953]
# order_ids = all_data_ml_df[all_data_ml_df['order_id'] > 108034]['order_id'].tolist()
order_ids = manual_check_df['order_id'].tolist()
order_id=109748
error_ids = []
for order_id in order_ids:
    print(order_id)
    df = get_order_data(order_id, is_sql=True)
    df = process_data_mibao(df)
    df = df[mibao_ml_features]
    base_df = all_data_ml_df[mibao_ml_features][all_data_ml_df['order_id'] == order_id]
    cmp_df = pd.concat([base_df, df])
    result = cmp_df.std().sum()
    if (result > 0):
        error_ids.append(order_id)
        print("error with oder_id {}".format(error_ids))
        break


print("final result {}".format(error_ids))


# In[]

#  检验在线预测与事后预测结果是否一致
# all_data_df = pd.read_csv(os.path.join(workdir, 'mibaodata_ml.csv'), encoding='utf-8', engine='python')
# result_df = pd.read_csv(os.path.join(workdir, 'mibao_mlresult.csv'), encoding='utf-8', engine='python')
# y_pred = lgb_clf.predict(x)
# result_df['pred_pickle'] = y_pred
# diff_df = result_df[result_df['predict'] != result_df['pred_pickle']]

# In[]
