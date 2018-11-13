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
from log import log
import random


warnings.filterwarnings('ignore')
workdir = get_workdir()

# 获取训练数据
all_data_df = pd.read_csv(os.path.join(workdir, "mibaodata_ml.csv"), encoding='utf-8', engine='python')
df = all_data_df.copy()
print("数据量: {}".format(df.shape))


# 模型测试：逐一读取数据库文件， 检验数据处理结果与机器学习处理结果是否一直
def model_test():
    # order_id =9085, 9098的crate_time 是错误的
    order_ids = random.sample(all_data_df['order_id'].tolist(), 100)
    order_ids = all_data_df[all_data_df['order_id']>1431]['order_id'].tolist()
    order_ids = [126, 127, 128, 140, 198, 223, 278, 284, 486, 492, 494, 558, 594, 878, 901]
    order_id = 126
    error_ids = []
    for order_id in order_ids:
        print(order_id)
        df = get_order_data(order_id, is_sql=True)
        processed_df = process_data_mibao(df.copy())
        cmp_df = pd.concat([all_data_df, processed_df])
        cmp_df = cmp_df[cmp_df['order_id'] == order_id]
        result = cmp_df.std().sum()
        if(result > 0):
            error_ids.append(order_id)
            print("error with oder_id {}".format(error_ids))

    pass

# model_test()


# In[]

# 测试预测能力
# all_data_df = pd.read_csv(os.path.join(workdir, 'mibaodata_ml.csv'), encoding='utf-8', engine='python')
# result_df = pd.read_csv(os.path.join(workdir, 'mibao_mlresult.csv'), encoding='utf-8', engine='python')
# y_pred = lgb_clf.predict(x)
# result_df['pred_pickle'] = y_pred
# diff_df = result_df[result_df['predict'] != result_df['pred_pickle']]

# In[]