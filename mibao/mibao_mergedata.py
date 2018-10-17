# coding: utf-8

import csv
import json

import pandas as pd
import numpy as np
import os
# Suppress warnings
import warnings
from mlutils import *
import operator

warnings.filterwarnings('ignore')
# to make output display better
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 2000)

# read large csv file
csv.field_size_limit(100000000)

PROJECT_ID = 'mibao'
# ## 获取数据
if os.getcwd().find(PROJECT_ID) == -1:
    os.chdir(PROJECT_ID)
datasets_path = os.getcwd() + '\\datasets\\'

# 读取并处理主表order, 所有表合并成all_data_df
df = pd.read_csv(datasets_path + "order.csv", encoding='utf-8', engine='python')
df = df[['id', 'create_time', 'merchant_id', 'user_id', 'state', 'cost', 'installment', 'pay_num',
         'added_service', 'bounds_example_id', 'bounds_example_no', 'goods_type', 'lease_term',
         'commented', 'accident_insurance', 'type', 'order_type', 'device_type', 'source', 'distance',
         'disposable_payment_discount', 'disposable_payment_enabled', 'lease_num', 'merchant_store_id',
         'deposit', 'hit_merchant_white_list', 'fingerprint', 'cancel_reason', 'delivery_way']]
df.rename(columns={'id': 'order_id'}, inplace=True)

# 根据state生成TARGET，代表最终审核是否通过
state_values = ['pending_receive_goods', 'running', 'user_canceled', 'pending_pay',
                'artificial_credit_check_unpass_canceled', 'pending_artificial_credit_check', 'lease_finished',
                'return_overdue', 'order_payment_overtime_canceled', 'pending_send_goods',
                'merchant_not_yet_send_canceled', 'running_overdue', 'buyout_finished', 'pending_user_compensate',
                'repairing', 'express_rejection_canceled', 'pending_return', 'returning', 'return_goods',
                'pending_relet_check', 'returned_received', 'relet_finished', 'merchant_relet_check_unpass_canceled',
                'system_credit_check_unpass_canceled', 'pending_jimi_credit_check', 'pending_relet_start',
                'pending_refund_deposit', 'merchant_credit_check_unpass_canceled']
failure_state_values = ['user_canceled', 'artificial_credit_check_unpass_canceled', 'return_overdue', 'running_overdue',
                        'merchant_relet_check_unpass_canceled', 'system_credit_check_unpass_canceled',
                        'merchant_credit_check_unpass_canceled']
pending_state_values = ['pending_artificial_credit_check', 'pending_relet_check', 'pending_jimi_credit_check',
                        'pending_relet_start']
state_values_newest = df['state'].unique().tolist()
# 若state字段有新的状态产生， 抛出异常
assert (operator.eq(state_values_newest, state_values))

df = df[df['state'].isin(pending_state_values + ['user_canceled']) != True]
df.insert(0, 'target', np.where(df['state'].isin(failure_state_values), 0, 1))

# 去除测试数据和内部员工数据
df = df[df['cancel_reason'].str.contains('测试') != True]
# df = df[df['check_remark'].str.contains('测试|内部员工') != True] #order_buyout表
# 去除命中商户白名单的订单
df = df[df['hit_merchant_white_list'].str.contains('01') != True]

df.drop(['cancel_reason', 'hit_merchant_white_list'], axis=1, inplace=True, errors='ignore')
all_data_df = df.copy()

# 读取并处理表user
df = pd.read_csv(datasets_path + "user.csv")
df = df[['id', 'head_image_url', 'recommend_code', 'regist_channel_type', 'share_callback', 'tag', 'phone']]
df.rename(columns={'id': 'user_id', 'phone': 'phone_user'}, inplace=True)
all_data_df = pd.merge(all_data_df, df, on='user_id', how='left')

# 读取并处理表 bargain_help
df = pd.read_csv(datasets_path + "bargain_help.csv")
all_data_df['have_bargain_help'] = np.where(all_data_df['user_id'].isin(df['user_id'].values), 1, 0)
# 读取并处理表 face_id
df = pd.read_csv(datasets_path + "face_id.csv")
df = df[['user_id', 'status']]
df.rename(columns={'status': 'face_check'}, inplace=True)
all_data_df = pd.merge(all_data_df, df, on='user_id', how='left')

# 读取并处理表 face_id_liveness
df = pd.read_csv(datasets_path + "face_id_liveness.csv")
df = df[['order_id', 'status']]
df.rename(columns={'status': 'face_live_check'}, inplace=True)
all_data_df = pd.merge(all_data_df, df, on='order_id', how='left')

# 读取并处理表 order_express
# 未处理特征：'country', 'provice', 'city', 'regoin', 'receive_address', 'live_address'
df = pd.read_csv(datasets_path + "order_express.csv")
df = df[['order_id', 'zmxy_score', 'card_id', 'phone', 'company', ]]
df.drop_duplicates(subset='order_id', inplace=True)
all_data_df = pd.merge(all_data_df, df, on='order_id', how='left')

# 读取并处理表 order_detail
df = pd.read_csv(datasets_path + "order_detail.csv")
df = df[['order_id', 'order_detail']]
df['xiaobaiScore'] = df['order_detail'].map(lambda x: json.loads(x).get('xiaobaiScore'))
df['zmxyScore'] = df['order_detail'].map(lambda x: json.loads(x).get('zmxyScore'))
df.drop(labels='order_detail', axis=1, inplace=True)
all_data_df = pd.merge(all_data_df, df, on='order_id', how='left')


# 读取并处理表 order_phone_book
def count_name_nums(data):
    data_list = json.loads(data)
    name_list = []
    for phone_book in data_list:
        if len(phone_book.get('name')) > 0 and phone_book.get('name').isdigit() is False:
            name_list.append(phone_book.get('name'))

    return len(set(name_list))


df = pd.read_csv(datasets_path + "order_phone_book.csv")
df = df[['order_id', 'emergency_contact_name', 'phone_book', 'emergency_contact_phone', 'emergency_contact_relation']]
df['phone_book'] = df['phone_book'].map(count_name_nums)
all_data_df = pd.merge(all_data_df, df, on='order_id', how='left')

# 读取并处理表 order_goods
# 未处理特征：
df = pd.read_csv(datasets_path + "order_goods.csv")
df = df[['order_id', 'num', 'price', 'category', 'old_level', ]]
df.drop_duplicates(subset='order_id', inplace=True)
all_data_df = pd.merge(all_data_df, df, on='order_id', how='left')

# 读取并处理表 order_xinyongzu
# 未处理特征：
# df = pd.read_csv(datasets_path + "order_xinyongzu.csv")
# df = df[['order_id', 'emergency_contact_name', 'phone_book', 'emergency_contact_phone', 'emergency_contact_relation']]
# df['phone_book'].str.count('mobile')
# all_data_df = pd.merge(all_data_df, df, on='order_id', how='left')

# 读取并处理表 risk_order
# 未处理特征：
df = pd.read_csv(datasets_path + "risk_order.csv")
df = df[['order_id', 'type', 'result', 'detail_json', ]]
df['result'] = df['result'].str.lower()

for risk_type in df['type'].unique().tolist():
    tmp_df = df[df['type'].str.match(risk_type)][['order_id', 'result', 'detail_json']]
    tmp_df.rename(
        columns={'result': risk_type + '_result', 'detail_json': risk_type + '_detail_json'},
        inplace=True)
    all_data_df = pd.merge(all_data_df, tmp_df, on='order_id', how='left')

# 读取并处理表 risk_white_list
# 未处理特征：
df = pd.read_csv(datasets_path + "risk_white_list.csv")
user_ids = df['user_id'].values
all_data_df = all_data_df[all_data_df['user_id'].isin(user_ids) != True]

# 读取并处理表 user_credit
# 未处理特征：
df = pd.read_csv(datasets_path + "user_credit.csv")
df = df[['user_id', 'cert_no', 'workplace', 'idcard_pros', 'occupational_identity_type', 'company_phone',
         'cert_no_expiry_date', 'cert_no_json', ]]
all_data_df = pd.merge(all_data_df, df, on='user_id', how='left')

# 读取并处理表 user_device
# 未处理特征：
df = pd.read_csv(datasets_path + "user_device.csv")
df = df[['user_id', 'device_type', 'regist_device_info', 'regist_useragent', 'ingress_type']]
df.rename(columns={'device_type': 'device_type_os'}, inplace=True)
all_data_df = pd.merge(all_data_df, df, on='user_id', how='left')

# 读取并处理表 user_third_party_account
# 未处理特征：
df = pd.read_csv(datasets_path + "user_third_party_account.csv")
counts_df = pd.DataFrame({'account_num': df['user_id'].value_counts()})
counts_df['user_id'] = counts_df.index
all_data_df = pd.merge(all_data_df, counts_df, on='user_id', how='left')

# 读取并处理表 user_zhima_cert
# 未处理特征：
df = pd.read_csv(datasets_path + "user_zhima_cert.csv")
df = df[['user_id', 'status', ]][df['status'].str.match('PASSED')]
all_data_df['zhima_cert_result'] = np.where(all_data_df['user_id'].isin(df['user_id'].tolist()), 1, 0)

# 读取并处理表 用户登陆使用情况， 在mango数据库
# df = pd.read_csv(datasets_path + "ser_login_record.csv")
# df = df[['user_id', 'final_score', 'final_decision', 'result_desc', 'supplement_info']]
# all_data_df = pd.merge(all_data_df, df, on='user_id', how='left')

'''
feature = 'input_parameter_json'
df[feature].value_counts()
df.shape
missing_values_table(df)
df[feature].unique()
df.columns.values
missing_values_table(all_data_df)
'''
# 保存数据
all_data_df.drop(['mibao_result'], axis=1, inplace=True, errors='ignore')
missing_values_table(all_data_df)
all_data_df.to_csv(datasets_path + "mibao.csv", index=False)
print("mibao.csv saved with shape {}".format(all_data_df.shape))
# In[]
exit('mergedata')
