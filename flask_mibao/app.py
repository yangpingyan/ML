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

# 初始化数据库连接，使用pymysql模块
engine = create_engine('mysql+pymysql://root:qawsedrf@localhost:3306/mibao')

app = Flask(__name__)
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir, 'mibao_ml.pkl'), 'rb'))

order_id = 88668
is_sql = True
cmp_df = all_data_df[all_data_df['order_id'] == order_id].copy()
test_df = get_order_data(order_id)
same_feature = list(set(cmp_df.columns.tolist() + test_df.columns.tolist()))

cmp_df  same_feature

def read_mlfile(filename, field='order_id', id_value=None, is_sql=False):
    if is_sql:
        sql = "SELECT * FROM `{}` o WHERE o.{} = {};".format(filename, field, id_value)
        df = pd.read_sql_query(sql, engine)
    else:
        df = pd.read_csv(datasets_path + filename + '.csv', encoding='utf-8', engine='python')
    return df


def get_order_data(order_id, is_sql=True):
    # list(df.loc[63648].values.tolist())

    # 读取order表
    order_df = read_mlfile('order', 'id', order_id, is_sql)
    order_df = order_df[['id', 'create_time', 'merchant_id', 'user_id', 'state', 'cost', 'installment', 'pay_num',
                         'added_service', 'bounds_example_id', 'bounds_example_no', 'goods_type', 'lease_term',
                         'commented', 'accident_insurance', 'type', 'order_type', 'device_type', 'source', 'distance',
                         'disposable_payment_discount', 'disposable_payment_enabled', 'lease_num', 'merchant_store_id',
                         'deposit', 'hit_merchant_white_list', 'fingerprint', 'cancel_reason', 'delivery_way',
                         'order_number']]
    order_df.rename(columns={'id': 'order_id'}, inplace=True)
    user_id = order_df.at[0, 'user_id']
    order_number = order_df.at[0, 'order_number']

    all_data_df = order_df.copy()

    # 读取并处理表 user
    user_df = read_mlfile('user', 'id', user_id, is_sql)
    user_df = user_df[
        ['id', 'head_image_url', 'recommend_code', 'regist_channel_type', 'share_callback', 'tag', 'phone']]
    user_df.rename(columns={'id': 'user_id', 'phone': 'phone_user'}, inplace=True)
    all_data_df = pd.merge(all_data_df, user_df, on='user_id', how='left')

    # 读取并处理表 bargain_help
    bargain_help_df = read_mlfile('bargain_help', 'user_id', user_id, is_sql)
    all_data_df['have_bargain_help'] = np.where(all_data_df['user_id'].isin(bargain_help_df['user_id'].values), 1, 0)

    # 读取并处理表 face_id
    face_id_df = read_mlfile('face_id', 'user_id', user_id, is_sql)
    face_id_df = face_id_df[['user_id', 'status']]
    face_id_df.rename(columns={'status': 'face_check'}, inplace=True)
    all_data_df = pd.merge(all_data_df, face_id_df, on='user_id', how='left')

    # 读取并处理表 face_id_liveness
    face_id_liveness_df = read_mlfile('face_id_liveness', 'order_id', order_id, is_sql)
    face_id_liveness_df = face_id_liveness_df[['order_id', 'status']]
    face_id_liveness_df.rename(columns={'status': 'face_live_check'}, inplace=True)
    all_data_df = pd.merge(all_data_df, face_id_liveness_df, on='order_id', how='left')

    # 读取并处理表 user_credit
    user_credit_df = read_mlfile('user_credit', 'user_id', user_id, is_sql)
    user_credit_df = user_credit_df[
        ['user_id', 'cert_no', 'workplace', 'idcard_pros', 'occupational_identity_type', 'company_phone',
         'cert_no_expiry_date', 'cert_no_json', ]]
    all_data_df = pd.merge(all_data_df, user_credit_df, on='user_id', how='left')

    # 读取并处理表 user_device
    user_device_df = read_mlfile('user_device', 'user_id', user_id, is_sql)
    user_device_df = user_device_df[
        ['user_id', 'device_type', 'regist_device_info', 'regist_useragent', 'ingress_type']]
    user_device_df.rename(columns={'device_type': 'device_type_os'}, inplace=True)
    all_data_df = pd.merge(all_data_df, user_device_df, on='user_id', how='left')

    # 读取并处理表 order_express
    # 未处理特征：'country', 'provice', 'city', 'regoin', 'receive_address', 'live_address'
    order_express_df = read_mlfile('order_express', 'order_id', order_id, is_sql)
    order_express_df = order_express_df[['order_id', 'zmxy_score', 'card_id', 'phone', 'company', ]]
    order_express_df.drop_duplicates(subset='order_id', inplace=True)
    all_data_df = pd.merge(all_data_df, order_express_df, on='order_id', how='left')

    # 读取并处理表 order_detail
    order_detail_df = read_mlfile('order_detail', 'order_id', order_id, is_sql)
    order_detail_df = order_detail_df[['order_id', 'order_detail']]
    all_data_df = pd.merge(all_data_df, order_detail_df, on='order_id', how='left')

    # 读取并处理表 order_goods
    order_goods_df = read_mlfile('order_goods', 'order_id', order_id, is_sql)
    order_goods_df = order_goods_df[['order_id', 'price', 'category', 'old_level', ]]
    order_goods_df.drop_duplicates(subset='order_id', inplace=True)
    all_data_df = pd.merge(all_data_df, order_goods_df, on='order_id', how='left')

    # 读取并处理表 order_phone_book
    order_phone_book_df = read_mlfile('order_phone_book', 'order_id', order_id, is_sql)
    order_phone_book_df = order_phone_book_df[['order_id', 'phone_book', ]]
    all_data_df = pd.merge(all_data_df, order_phone_book_df, on='order_id', how='left')

    # 读取并处理表 risk_order
    risk_order_df = read_mlfile('risk_order', 'order_id', order_id, is_sql)
    risk_order_df = risk_order_df[['order_id', 'type', 'result', 'detail_json', ]]
    risk_order_df['result'] = risk_order_df['result'].str.lower()
    for risk_type in risk_order_df['type'].unique().tolist():
        tmp_df = risk_order_df[risk_order_df['type'].str.match(risk_type)][['order_id', 'result', 'detail_json']]
        tmp_df.rename(
            columns={'result': risk_type + '_result', 'detail_json': risk_type + '_detail_json'},
            inplace=True)
        all_data_df = pd.merge(all_data_df, tmp_df, on='order_id', how='left')
    # 读取并处理表 tongdun
    tongdun_df = read_mlfile('tongdun', 'order_number', order_number, is_sql)
    tongdun_df = tongdun_df[['order_number', 'final_score', 'final_decision']]
    all_data_df = pd.merge(all_data_df, tongdun_df, on='order_number', how='left')

    # 读取并处理表 user_third_party_account
    user_third_party_account_df = read_mlfile('user_third_party_account', 'user_id', user_id, is_sql)
    counts_df = pd.DataFrame({'account_num': user_third_party_account_df['user_id'].value_counts()})
    counts_df['user_id'] = counts_df.index
    all_data_df = pd.merge(all_data_df, counts_df, on='user_id', how='left')

    # 读取并处理表 user_zhima_cert
    user_zhima_cert_df = read_mlfile('user_zhima_cert', 'user_id', user_id, is_sql)
    user_zhima_cert_df = user_zhima_cert_df[['user_id', 'status', ]][user_zhima_cert_df['status'].str.match('PASSED')]
    all_data_df['zhima_cert_result'] = np.where(all_data_df['user_id'].isin(user_zhima_cert_df['user_id'].tolist()), 1,
                                                0)

    # 读取并处理表 jimi_order_check_result_list
    jimi_order_check_result_df = read_mlfile('jimi_order_check_result', 'order_id', order_id, is_sql)
    jimi_order_check_result_df = jimi_order_check_result_df[['order_id', 'check_remark']]
    all_data_df = pd.merge(all_data_df, jimi_order_check_result_df, on='order_id', how='left')

    # 丢弃不需要的数据
    # 去掉白名单用户 读取并处理表 risk_white_list
    risk_white_list_df = read_mlfile('risk_white_list', 'user_id', user_id, is_sql)
    user_ids = risk_white_list_df['user_id'].values
    all_data_df = all_data_df[all_data_df['user_id'].isin(user_ids) != True]
    if is_sql:
        if len(risk_white_list_df) > 0:
            print("命中白名单用户： order_id({}), user_id({})".format(order_id, user_id))

    if is_sql == False:
        # 根据state生成target，代表最终审核是否通过
        state_values = ['pending_receive_goods', 'running', 'user_canceled', 'pending_pay',
                        'artificial_credit_check_unpass_canceled', 'pending_artificial_credit_check', 'lease_finished',
                        'return_overdue', 'order_payment_overtime_canceled', 'pending_send_goods',
                        'merchant_not_yet_send_canceled', 'running_overdue', 'buyout_finished', 'pending_user_compensate',
                        'repairing', 'express_rejection_canceled', 'pending_return', 'returning', 'return_goods',
                        'pending_relet_check', 'returned_received', 'relet_finished',
                        'merchant_relet_check_unpass_canceled',
                        'system_credit_check_unpass_canceled', 'pending_jimi_credit_check', 'pending_relet_start',
                        'pending_refund_deposit', 'merchant_credit_check_unpass_canceled']
        failure_state_values = ['user_canceled', 'artificial_credit_check_unpass_canceled', 'return_overdue',
                                'running_overdue',
                                'merchant_relet_check_unpass_canceled', 'system_credit_check_unpass_canceled',
                                'merchant_credit_check_unpass_canceled']
        pending_state_values = ['pending_artificial_credit_check', 'pending_relet_check', 'pending_jimi_credit_check',
                                'pending_relet_start']
        state_values_newest = all_data_df['state'].unique().tolist()
        # 若state字段有新的状态产生， 抛出异常
        assert (len(list(set(state_values_newest).difference(set(state_values)))) == 0)
        len(state_values_newest)
        len(state_values)
        all_data_df = all_data_df[all_data_df['state'].isin(pending_state_values + ['user_canceled']) != True]
        all_data_df.insert(0, 'target', np.where(all_data_df['state'].isin(failure_state_values), 0, 1))

        # 去除测试数据和内部员工数据
        all_data_df = all_data_df[all_data_df['cancel_reason'].str.contains('测试') != True]
        all_data_df = all_data_df[all_data_df['check_remark'].str.contains('测试') != True]
        # 去除命中商户白名单的订单
        all_data_df = all_data_df[all_data_df['hit_merchant_white_list'].str.contains('01') != True]

    all_data_df.drop(['mibao_result', 'order_number', 'cancel_reason', 'hit_merchant_white_list', 'check_remark'],
                     axis=1,
                     inplace=True, errors='ignore')

    return all_data_df


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
