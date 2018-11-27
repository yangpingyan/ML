# coding:utf8
import os
import time
from argparse import ArgumentParser
from flask import Flask, jsonify
from flask import make_response
import pandas as pd
import json
import lightgbm as lgb
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
import warnings
from gevent.pywsgi import WSGIServer
from mltools import *
from mldata import *
import logging
from mibao_log import log
import random
from explore_data_utils import add_score

log.debug(time.asctime())
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 60)

# 获取训练数据
df = pd.read_csv(os.path.join(workdir, "mibaodata_ml.csv"), encoding='utf-8', engine='python')
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
score_df = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'])
add_score(score_df, 'auc', y_test, y_pred)
log.debug(score_df)
accuracy_score = accuracy_score(y_test, y_pred)

assert accuracy_score > 0.96

# In[]
'''
auc score: 0.9737308622078968 0.9496567505720824
auc score: 0.9711522965350524 0.942351598173516
auc score: 0.9701853344077357 0.93190770962296
auc score: 0.9727639000805802 0.9418079096045198
auc score: 0.972119258662369 0.9441371681415929


'''

app = Flask(__name__)

# start with order_id 105914
@app.route('/ml_result/<int:order_id>', methods=['GET'])
def get_predict_result(order_id):
    # log.debug("order_id: {}".format(order_id))
    ret_data = 2
    df = get_order_data(order_id, is_sql=True)
    if len(df) != 0:
        log.debug(df[['order_id', 'state', 'state_cao']])
        df = process_data_mibao(df)
        df = df[mibao_ml_features]
        # print(len(df.columns))
        # print(list(set(all_data_df.columns.tolist()).difference(set(df.columns.tolist()))))
        if len(df.columns) == 53:
            y_pred = lgb_clf.predict(df)
            ret_data = y_pred[0]
    log.debug("order_id {} result: {}".format(order_id, ret_data))
    # print("reference:", all_data_df[all_data_df['order_id'] == order_id])
    return jsonify({"code": 200, "data": {"result": int(ret_data)}, "message": "SUCCESS"}), 200


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

