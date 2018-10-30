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

app = Flask(__name__)
clf = pickle.load(open('mibao_ml.pkl', 'rb'))
y_pred = clf.predict(x_test)
add_score(score_df, 'app', y_pred, y_test)

users = [
            {
                'id': 1,
                'title': u'Buy groceries',
                'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
                'done': False
            },
            {
                'id': 2,
                'title': u'Learn Python',
                'description': u'Need to find a good Python tutorial on the web',
                'done': False
            }
        ] \

        @ app.route('/ml/<int:user_id>', methods=['GET'])


def get_result(user_id):
    print(user_id)
    # result = filter(lambda u: u['id'] == user_id, users)
    # print(result)
    # # if len(result) == 0:
    # #     abort(404)
    return jsonify({'users': users[0]})


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
