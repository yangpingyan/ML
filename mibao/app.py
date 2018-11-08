# coding:utf8
import os
import time
from argparse import ArgumentParser
from flask import Flask, request, jsonify
from flask import make_response
import pandas as pd
import json
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlutils
import warnings
from gevent.pywsgi import WSGIServer

warnings.filterwarnings('ignore')

def model_test():
    pass
def get_workdir(projectid):
    try:
        cur_dir = os.path.dirname(__file__)
    except:
        cur_dir = os.getcwd()
    if cur_dir.find(projectid) == -1:
        cur_dir = os.path.join(cur_dir, projectid)
    return cur_dir


# 设置随机种子
# np.random.seed(88)
# ## 获取数据
PROJECT_ID = 'mibao'
workdir = get_workdir(PROJECT_ID)
df = pd.read_csv(os.path.join(workdir, "mibaodata_ml.csv"), encoding='utf-8', engine='python')
print("数据量: {}".format(df.shape))
x = df.drop(['target', 'order_id'], axis=1)
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# Create a training and testing dataset
# train_set = lgb.Dataset(data=x_train, label=y_train)
# test_set = lgb.Dataset(data=x_test, label=y_test)

with open('lgb_params.json', 'r') as f:
    lgb_params_auc = json.load(f)

lgb_clf = lgb.LGBMClassifier(**lgb_params_auc)
lgb_clf.fit(x_train, y_train)
y_pred = lgb_clf.predict(x_test)
accuracy_score = accuracy_score(y_test, y_pred)
print("auc score:", accuracy_score)
assert accuracy_score>0.96
# 测试预测能力
# all_data_df = pd.read_csv(os.path.join(workdir, 'mibaodata_ml.csv'), encoding='utf-8', engine='python')
# result_df = pd.read_csv(os.path.join(workdir, 'mibao_mlresult.csv'), encoding='utf-8', engine='python')
# y_pred = lgb_clf.predict(x)
# result_df['pred_pickle'] = y_pred
# diff_df = result_df[result_df['predict'] != result_df['pred_pickle']]

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
app = Flask(__name__)


@app.route('/ml/<int:order_id>', methods=['GET'])
def get_predict_result(order_id):
    starttime = time.clock()
    df = mlutils.get_order_data(order_id, is_sql=True)
    if len(df) == 0:
        return make_response(jsonify({'error': 'order_id error'}), 403)
    df = mlutils.process_data_mibao(df)

    df.drop(['order_id'], axis=1, inplace=True, errors='ignore')
    # print(list(set(all_data_df.columns.tolist()).difference(set(df.columns.tolist()))))

    if len(df.columns) != 54:
        return make_response(jsonify({'error': 'data error'}), 404)
    # order_data = np.array(df).reshape(1, -1)
    y_pred = lgb_clf.predict(df)
    print("y_pred:", y_pred[0])
    # print("reference:", result_df[result_df['order_id'] == order_id])
    return jsonify({'ml_result': int(y_pred[0])}), 201


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
