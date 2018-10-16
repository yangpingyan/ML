# coding: utf-8

import csv
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
# 未处理feature: ip,
df = pd.read_csv(datasets_path + "order.csv", encoding='utf-8', engine='python')
df = df[['id', 'create_time', 'merchant_id', 'user_id', 'state', 'cost', 'installment', 'pay_num',
         'added_service', 'bounds_example_id', 'bounds_example_no', 'goods_type', 'lease_term',
         'commented', 'accident_insurance', 'type', 'order_type', 'device_type', 'source', 'distance',
         'disposable_payment_discount', 'disposable_payment_enabled', 'lease_num', 'merchant_store_id',
         'deposit', 'hit_merchant_white_list', 'fingerprint', 'cancel_reason', 'releted']]
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
df.drop(['state', 'cancel_reason'], axis=1, inplace=True, errors='ignore')
all_data_df = df.copy()

# 读取并处理表user
df = pd.read_csv(datasets_path + "user.csv")
df = df[['id', 'head_image_url', 'recommend_code', 'regist_channel_type', 'share_callback', 'tag']]
df.rename(columns={'id': 'user_id'}, inplace=True)
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
# 处理芝麻信用分 '>600' 更改成600
zmf = [0] * len(df)
xbf = [0] * len(df)
for row, detail in enumerate(df['zmxy_score']):
    # print(row, detail)
    if isinstance(detail, str):
        if '/' in detail:
            score = detail.split('/')
            xbf[row] = 0 if score[0] == '' else (float(score[0]))
            zmf[row] = 0 if score[1] == '' else (float(score[1]))
        # print(score, row)
        elif '>' in detail:
            zmf[row] = 600
        else:
            score = float(detail)
            if score <= 200:
                xbf[row] = (score)
            else:
                zmf[row] = (score)

df['zmf'] = zmf
df['xbf'] = xbf
zmf_most = df['zmf'][df['zmf'] > 0].value_counts().index[0]
xbf_most = df['xbf'][df['xbf'] > 0].value_counts().index[0]
df['zmf'][df['zmf'] == 0] = zmf_most
df['xbf'][df['xbf'] == 0] = xbf_most
# 根据身份证号增加性别和年龄 年龄的计算需根据订单创建日期计算
df['age'] = df['card_id'].map(lambda x: 2018 - int(x[6:10]))
df['sex'] = df['card_id'].map(lambda x: int(x[-2]) % 2)
df['phone'] = df['phone'].astype(str)
df['phone'][df['phone'].str.len() != 11] = '0'
df['phone'] = df['phone'].str.slice(0, 3)
df.drop(labels=['card_id', 'zmxy_score'], axis=1, inplace=True, errors='ignore')
all_data_df = pd.merge(all_data_df, df, on='order_id', how='left')
# 读取并处理表 order_phone_book
# 未处理特征：
df = pd.read_csv(datasets_path + "order_phone_book.csv")
df = df[['order_id', 'emergency_contact_name', 'phone_book', 'emergency_contact_phone', 'emergency_contact_relation']]
df['phone_book'] = df['phone_book'].str.count('mobile')
all_data_df = pd.merge(all_data_df, df, on='order_id', how='left')

# 读取并处理表 order_goods
# 未处理特征：
df = pd.read_csv(datasets_path + "order_goods.csv")
df = df[['order_id', 'num', 'price', 'category', 'old_level', ]]
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
    tmp_df = df[df['type'].str.match(risk_type)]
    tmp_df.rename(
        columns={'type': risk_type + '_type', 'result': risk_type + '_result',
                 'detail_json': risk_type + '_detail_json'},
        inplace=True)
    all_data_df = pd.merge(all_data_df, tmp_df, on='order_id', how='left')

# 读取并处理表 risk_white_list
# 未处理特征：
df = pd.read_csv(datasets_path + "risk_white_list.csv")
user_ids = df['user_id'].values
all_data_df = all_data_df[all_data_df['user_id'].isin(user_ids) != True]

# ser_login_record todo
# user_longitude_latitude todo

# 读取并处理表 merchant_white_list todo
# 未处理特征：
df = pd.read_csv(datasets_path + "merchant_white_list.csv")
user_ids = df['user_id'].values
# all_data_df = all_data_df[all_data_df['user_id'].isin(user_ids) != True]

# 读取并处理表 tongdun todo
# 未处理特征：
df = pd.read_csv(datasets_path + "tongdun.csv")

# 读取并处理表 user_active
# 未处理特征：
df = pd.read_csv(datasets_path + "user_credit.csv")
df = df[['user_id', 'cert_no', 'workplace', 'idcard_pros', 'idcard_cons', 'handheld_id_card', 'zhima_score',
         'occupational_identity_type', 'company_phone', 'cert_no_expiry_date', 'cert_no_json', ]]
all_data_df = pd.merge(all_data_df, df, on='user_id', how='left')

# 读取并处理表 user_device
# 未处理特征：
df = pd.read_csv(datasets_path + "user_device.csv")
df = df[['user_id', 'device_type', 'regist_device_info', 'regist_useragent', 'ingress_type'  ]]
all_data_df = pd.merge(all_data_df, df, on='user_id', how='left')

# 读取并处理表 user_third_party_account
# 未处理特征：
df = pd.read_csv(datasets_path + "user_third_party_account.csv")
df = df[['user_id', 'gender', 'user_type',  ]]
all_data_df = pd.merge(all_data_df, df, on='user_id', how='left')

# 读取并处理表 user_zhima_cert
# 未处理特征：
df = pd.read_csv(datasets_path + "user_zhima_cert.csv")
df = df[['user_id', 'status', ]]
df.rename(columns={'status': 'zhima_cert_result'}, inplace=True)
all_data_df = pd.merge(all_data_df, df, on='user_id', how='left')


'''
feature = 'status'
df[feature].value_counts()
df.shape
missing_values_table(df)
df[feature].unique()
df.columns.values
missing_values_table(all_data_df)
'''

# 保存数据
missing_values_table(all_data_df)
all_data_df.to_csv(datasets_path + "mibao.csv", index=False)
exit('merge')

'''
feature = 'result'
df[feature].value_counts()
df[feature].fillna(value=-999, inplace=True)
feature_analyse(df, feature)
df[df[feature].isnull()].sort_values(by='state').shape
df.shape
missing_values_table(df)
df[feature].unique()
df.columns.values
missing_values_table(all_data_df)
'''
# read user data


all_data_df.to_csv(datasets_path + "mibao.csv", index=False)
print("mibao.csv saved")

# merchant 违约率
'''
df.columns.values
feature = 'have_bargain_help'
df[feature].value_counts()
feature_analyse(df, feature)
df[df[feature].isnull()].sort_values(by='state').shape

missing_values_table(df)
df[feature].unique()
'''
df.drop(['user_id', ], axis=1, inplace=True, errors='ignore')

datasets_path
print("保存的数据量: {}".format(df.shape))
df.to_csv(datasets_path + "mibaodata_ml.csv", index=False)

# 查看各特征关联度
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df.astype(float).corr(), linewidths=0.1, vmax=1.0,
            square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)
plt.show()

'''
调试代码
df.sort_values(by=['device_type'], inplace=True, axis=1)
df['device_type'].value_counts()
df['device_type'].fillna(value='NODATA', inplace=True)
feature_analyse(df, 'bounds_example_id')
df.columns.values
missing_values_table(df)
'''
'''
现有数据特征挖掘（与客户和订单信息相关的数据都可以拿来做特征）：
1. 同盾和白骑士中的内容详情
2. IP地址、收货地址、身份证地址、居住地址中的关联
3. 优惠券，额外服务意外险
4. 首付金额、每日租金
5. 手机号前三位等
6. 下单使用的设备，通过什么客户端下单（微信、支付宝、京东、网页）
7. 是否有推荐人， 推荐人是否通过审核
#. 租借个数

 b. 创造未被保存到数据库中的特征：年化利率,是否有2个手机号。
'''

# 所有字符串变成大写字母
objs_df = pd.DataFrame({"isobj": pd.Series(df.dtypes == 'object')})
df[objs_df[objs_df['isobj'] == True].index.values] = df[objs_df[objs_df['isobj'] == True].index.values].applymap(
    lambda x: x.upper() if isinstance(x, str) else x)

# 有phone_book的赋值成1， 空的赋值成0
df['phone_book'][df['phone_book'].notnull()] = 1
df['phone_book'][df['phone_book'].isnull()] = 0

# 同盾白骑士审核结果统一
df['result'] = df['result'].map(lambda x: x.upper() if isinstance(x, str) else 'NODATA')
df['result'][df['result'].str.match('ACCEPT')] = 'PASS'
# 有emergency_contact_phone的赋值成1， 空的赋值成0
df['emergency_contact_phone'][df['emergency_contact_phone'].notnull()] = 1
df['emergency_contact_phone'][df['emergency_contact_phone'].isnull()] = 0

features_cat = ['check_result', 'result', 'pay_num', 'channel', 'goods_type', 'lease_term', 'type', 'order_type',
                'source', 'phone_book', 'emergency_contact_phone', 'old_level', 'create_hour', 'sex', ]
features_number = ['cost', 'daily_rent', 'price', 'age', 'zmf_score', 'xbf_score', ]

for col in df[features_cat + features_number].columns.values:
    if df[col].dtype == 'O':
        df[col].fillna(value='NODATA', inplace=True)
        df.fillna(value=0, inplace=True)

plt.hist(df['check_result'])
feature_analyse(df, 'result')
feature_analyse(df, 'pay_num')
feature_analyse(df, 'channel')

feature = 'zmf_score'
plt.hist(df[feature])
feature_analyse(df, feature)
feature_kdeplot(df, feature)

# 芝麻分分类
bins = pd.IntervalIndex.from_tuples([(0, 600), (600, 700), (700, 800), (800, 1000)])
df['zmf_score_band'] = pd.cut(df['zmf_score'], bins, labels=False)
df[['zmf_score_band', 'check_result']].groupby(['zmf_score_band'], as_index=False).mean().sort_values(by='check_result',
                                                                                                      ascending=False)

# 小白分分类
bins = pd.IntervalIndex.from_tuples([(0, 80), (80, 90), (90, 100), (100, 200)])
df['xbf_score_band'] = pd.cut(df['xbf_score'], bins, labels=False)
df[['xbf_score_band', 'check_result']].groupby(['xbf_score_band'], as_index=False).mean().sort_values(by='check_result',
                                                                                                      ascending=False)

# 年龄分类
bins = pd.IntervalIndex.from_tuples([(0, 18), (18, 24), (24, 30), (30, 40), (40, 100)])
df['age_band'] = pd.cut(df['age'], bins, labels=False)
df[['age_band', 'check_result']].groupby(['age_band'], as_index=False).mean().sort_values(by='check_result',
                                                                                          ascending=False)

# 下单时间分类
df['create_hour_band'] = pd.cut(df['create_hour'], 5, labels=False)
df[['create_hour_band', 'check_result']].groupby(['create_hour_band'], as_index=False).mean().sort_values(
    by='check_result', ascending=False)

features = ['check_result', 'result', 'pay_num', 'channel', 'goods_type', 'type', 'order_type',
            'source', 'phone_book', 'old_level', 'sex', 'create_hour', 'age_band', 'zmf_score_band',
            'xbf_score_band', ]
df = df[features]
# 类别特征全部转换成数字
for feature in features:
    df[feature] = LabelEncoder().fit_transform(df[feature])

# ## 评估结果
# accuracy： 97.5%  --- 预测正确的个数占样本总数的比率
# precision： 91.4% --- 预测通过正确的个数占预测通过的比率
# recall：81.4% --- 预测通过正确的个数占实际通过的比率

# In[42]:


# 使用PR曲线： 当正例较少或者关注假正例多假反例。 其他情况用ROC曲线
plt.figure(figsize=(8, 6))
plt.xlabel("Recall(FPR)", fontsize=16)
plt.ylabel("Precision(TPR)", fontsize=16)
plt.axis([0, 1, 0, 1])

clf = rnd_clf
y_train_pred = cross_val_predict(clf, x_train, y_train, cv=3)
y_probas = cross_val_predict(clf, x_train, y_train, cv=3, method="predict_proba", n_jobs=-1)
y_scores = y_probas[:, 1]  # score = proba of positive class
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
plt.plot(recalls, precisions, linewidth=1, label="PR")
fpr, tpr, thresholds = roc_curve(y_train, y_scores)
print("{} roc socore: {}".format(clf.__class__.__name__, roc_auc_score(y_train, y_scores)))
plt.plot(fpr, tpr, linewidth=1, label="ROC")

plt.title("ROC and PR 曲线图")
plt.legend()
plt.show()

# ## 总结
# 1. 机器学习能洞察复杂问题和大量数据，发现内在规律，帮助我们做好数据挖掘方面的工作。
# 2. 很有很多的改进空间：
#         a.现有数据特征挖掘，譬如同盾和白骑士中的内容详情，IP地址、收货地址、身份证地址、居住地址中的关联，优惠券，额外服务意外险，每日租金，手机号前三位等。与客户和订单信息相关的数据都可以拿来做特征，提高机器学习预测能力。
#         b. 增加未被保存到数据库中的特征。
#         c. 调整机器学习参数，使之更适合做预测分类。
#         d. 使用更好的模型。XGBoosting、LightGBM和人工神经网络在训练和预测能力上做到更好。
# 3. 工作计划：
#         a. 增加特征。需数据库里保存所有订单审核前的所有信息。目前折扣金额和额外服务等数据只有再审核通过后才保存，需更改下。
#         b. 参考同类型行业，借鉴他人经验，增加相关特征
#         c. 增加模型预测能力，ROC分数达到0.96以上， 预测准确度达到98.5%
#         d. 增加客户信用额度字段。如何确定客户额度方案未知。
#
# ## 特征处理
# 1. 把IP地址、收货地址、身份证地址、居住地址转换成经纬度
# 2. 把创建时间转换成月、日、周几、小时段

# ## 增加特征
# 1. 下单时的经纬度
# 2. 下单时手机设备
# 3.
