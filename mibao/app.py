# coding:utf8
import re
import os
import time
import datetime
from argparse import ArgumentParser
from importlib import reload

import pytesseract
from PIL import Image
from flask import Flask, request, jsonify
from flask import make_response
from flask import abort
import pickle
import pandas as pd
import numpy as np
import json
import mlutils
reload(mlutils)

app = Flask(__name__)
cur_dir = mlutils.get_workdir('mibao')
clf = pickle.load(open(os.path.join(cur_dir, 'mibao_ml.pkl'), 'rb'))

all_data_df = pd.read_csv(os.path.join(cur_dir, 'datasets', 'mibaodata_ml.csv'), encoding='utf-8', engine='python')
all_data_df.drop('target', inplace=True, axis=1)

order_id = 88668
df = mlutils.get_order_data(order_id, is_sql=True)
df = mlutils.process_data_mibao(df)
all_data_df = pd.concat([all_data_df, df] )
all_data_df[all_data_df['order_id'] ==order_id]


@app.route('/ml/<int:order_id>', methods=['GET'])
def get_predict_result(order_id):
    print("order_id:", order_id)
    df = mlutils.get_order_data(order_id=88668, is_sql=True)
    df = mlutils.process_data_mibao(df)
    df.drop(['tongdun_detail_json', 'mibao_result', 'order_number', 'cancel_reason', 'hit_merchant_white_list',
                      'check_remark', 'cancel_reason', 'hit_merchant_white_list'], axis=1,
                     inplace=True, errors='ignore')
    if len(df.columns) != 56:
        return make_response(jsonify({'error': 'data error'}), 404)
    # order_data = np.array(df).reshape(1, -1)
    y_pred = clf.predict(df)
    print("y_pred:", y_pred[0])
    return jsonify({'ml_result': int(y_pred[0])}), 201


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
    opt = ArgumentParser()
    opt.add_argument('--model', default='gevent')
    args = opt.parse_args()
    if args.model == 'gevent':
        from gevent.pywsgi import WSGIServer

        http_server = WSGIServer(('0.0.0.0', 5000), app)
        print('listen on 0.0.0.0:5000')
        http_server.serve_forever()
    elif args.model == 'raw':
        app.run(host='0.0.0.0')
