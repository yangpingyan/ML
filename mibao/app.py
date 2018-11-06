# coding:utf8
import re
import os
import time
import datetime
from argparse import ArgumentParser
import pytesseract
from PIL import Image
from flask import Flask, request, jsonify
from flask import make_response
from flask import abort
import pickle
import pandas as pd
import numpy as np
import json
from mlutils import *



app = Flask(__name__)
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir, 'mibao_ml.pkl'), 'rb'))
# In[]
order_id = 88668
is_sql = True
features = ['user_id', 'status']
filename = 'order'
table = 'order_id'
id_value = 11

starttime = time.clock()
all_data_df = get_order_data(order_id=88668, is_sql=True)
print(time.clock()-starttime)

# In[]
@app.route('/ml/<int:order_id>', methods=['GET'])
def get_predict_result(order_id):
    print("order_id:", order_id)
    order_data = get_order_data(order_id)
    if len(order_data) != 59:
        return make_response(jsonify({'error': 'data error'}), 404)
    order_data = np.array(order_data).reshape(1, -1)
    y_pred = clf.predict(order_data)
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
