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

log.setLevel(logging.INFO)
warnings.filterwarnings('ignore')
workdir = get_workdir()

# 获取训练数据
all_data_df = pd.read_csv(os.path.join(workdir, "mibaodata_ml.csv"), encoding='utf-8', engine='python')
df = all_data_df.copy()
print("数据量: {}".format(df.shape))
x = df.drop(['target', 'order_id'], axis=1)
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# 机器学习模型训练
with open(os.path.join(workdir, "lgb_params.json"), 'r') as f:
    lgb_params_auc = json.load(f)

lgb_clf = lgb.LGBMClassifier(**lgb_params_auc)
lgb_clf.fit(x_train, y_train)
y_pred = lgb_clf.predict(x_test)
accuracy_score = accuracy_score(y_test, y_pred)
print("auc score:", accuracy_score)
assert accuracy_score>0.96

# In[]
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

model_test()
# 测试预测能力
# all_data_df = pd.read_csv(os.path.join(workdir, 'mibaodata_ml.csv'), encoding='utf-8', engine='python')
# result_df = pd.read_csv(os.path.join(workdir, 'mibao_mlresult.csv'), encoding='utf-8', engine='python')
# y_pred = lgb_clf.predict(x)
# result_df['pred_pickle'] = y_pred
# diff_df = result_df[result_df['predict'] != result_df['pred_pickle']]

# In[]

app = Flask(__name__)


@app.route('/ml/<int:order_id>', methods=['GET'])
def get_predict_result(order_id):
    df = get_order_data(order_id, is_sql=True)
    if len(df) == 0:
        return make_response(jsonify({'error': 'order_id error'}), 403)
    df = process_data_mibao(df)

    df.drop(['order_id'], axis=1, inplace=True, errors='ignore')
    # print(list(set(all_data_df.columns.tolist()).difference(set(df.columns.tolist()))))

    if len(df.columns) != 54:
        return make_response(jsonify({'error': 'data error'}), 404)
    # order_data = np.array(df).reshape(1, -1)
    y_pred = lgb_clf.predict(df)
    print("y_pred:", y_pred[0])
    print("reference:", all_data_df[all_data_df['order_id'] == order_id])
    return jsonify({'ml_result': int(y_pred[0])}), 201


@app.route('/debug/<int:debug>', methods=['GET'])
def set_debug_mode(debug):
    if debug == 1:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    log.debug("log mode ", log.level)
    return jsonify({'log_mode': int(log.level)}), 201

@app.errorhandler(404)
def not_found():
    return make_response(jsonify({'error': 'bug'}), 404)


if __name__ == '__main__':
    opt = ArgumentParser()
    opt.add_argument('--model', default='gevent')
    args = opt.parse_args()

    if args.model == 'gevent':
        http_server = WSGIServer(('0.0.0.0', 5000), app)
        print('listen on 0.0.0.0:5000')
        http_server.serve_forever()
    elif args.model == 'raw':
        app.run(host='0.0.0.0')

# In[]
