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
import warnings
import random
from sklearn.externals import joblib

warnings.filterwarnings('ignore')


# reload(mlutils)

def get_workdir(projectid):
    try:
        cur_dir = os.path.dirname(__file__)
    except:
        cur_dir = os.getcwd()
    if cur_dir.find(projectid) == -1:
        cur_dir = os.path.join(cur_dir, projectid)
    return cur_dir


app = Flask(__name__)
cur_dir = get_workdir('mibao')
clf = pickle.load(open(os.path.join(cur_dir, 'mibao_ml.pkl'), 'rb'))
clf = joblib.load(filename="mibao_ml.gz")
all_data_df = pd.read_csv(os.path.join(cur_dir, 'datasets', 'mibaodata_ml.csv'), encoding='utf-8', engine='python')
result_df = pd.read_csv(os.path.join(cur_dir, 'datasets', 'mibao_mlresult.csv'), encoding='utf-8', engine='python')

x = all_data_df.drop(['target', 'order_id'], axis=1, errors='ignore')
y_pred = clf.predict(x)
result_df['pred_pickle'] = y_pred
# In[]
'''
all_data_df = pd.read_csv(os.path.join(cur_dir, 'datasets', 'mibaodata_ml.csv'), encoding='utf-8', engine='python')

all_data_df.drop(['target'], inplace=True, axis=1, errors='ignore')
# In[]

order_ids = random.sample(all_data_df['order_id'].tolist(), 100)
# order_ids = [68050]
for order_id in order_ids:
    print(order_id)
    df = mlutils.get_order_data(order_id, is_sql=True)
    df = mlutils.process_data_mibao(df)
    cmp_df = pd.concat([all_data_df, df])
    cmp_df = cmp_df[cmp_df['order_id'] == order_id]
    result = cmp_df.std().sum()
    if(result > 0):
        print("error with oder_id {}".format(order_id))
'''


@app.route('/ml/<int:order_id>', methods=['GET'])
def get_predict_result(order_id):
    print("order_id:", order_id)
    df = mlutils.get_order_data(order_id, is_sql=True)
    print(df)
    if len(df) == 0:
        return make_response(jsonify({'error': 'order_id error'}), 403)
    df = mlutils.process_data_mibao(df)
    df.drop(['order_id'], axis=1, inplace=True, errors='ignore')
    if len(df.columns) != 54:
        return make_response(jsonify({'error': 'data error'}), 404)
    # order_data = np.array(df).reshape(1, -1)
    y_pred = clf.predict(df)
    print("y_pred:",  y_pred[0])
    print("reference:", result_df[result_df['order_id'] == order_id])
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

# In[]
