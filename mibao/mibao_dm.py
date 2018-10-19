# coding: utf-8

import csv
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
import warnings
from mlutils import *
import featuretools as ft
import json

# Suppress warnings
warnings.filterwarnings('ignore')
# to make output display better
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 2000)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['font.sans-serif'] = ['Simhei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# read large csv file
csv.field_size_limit(100000000)

PROJECT_ID = 'mibao'
# ## 获取数据
if os.getcwd().find(PROJECT_ID) == -1:
    os.chdir(PROJECT_ID)
datasets_path = os.getcwd() + '\\datasets\\'
all_data_df = pd.read_csv(datasets_path + "mibao.csv", encoding='utf-8', engine='python')
# In[]
df = all_data_df.copy()
print("data shape {}".format(all_data_df.shape))

# 开始处理特征
# 类别特征处理
features_cat = ['installment', 'commented', 'type', 'source', 'disposable_payment_enabled', 'merchant_store_id',
                'device_type', 'goods_type', 'merchant_id', 'order_type', 'regist_channel_type', 'face_check',
                'face_live_check', 'occupational_identity_type', 'ingress_type', 'device_type_os', 'bai_qi_shi_result',
                'guanzhu_result', 'tongdun_result', 'delivery_way', 'old_level', 'category', ]
for feature in features_cat:
    df[feature].fillna(value='NODATA' if df[feature].dtype == 'O' else -999, inplace=True)
    df[feature] = LabelEncoder().fit_transform(df[feature])

# 只判断是否空值的特征处理
features_cat_null = ['bounds_example_id', 'bounds_example_no', 'distance', 'fingerprint', 'added_service',
                     'recommend_code', 'regist_device_info', 'company', 'company_phone', 'workplace',
                     'emergency_contact_name', 'emergency_contact_phone', 'emergency_contact_relation',
                     'idcard_pros', ]
for feature in features_cat_null:
    df[feature] = np.where(df[feature].isnull(), 0, 1)

df['deposit'] = np.where(df['deposit'] == 0, 0, 1)

df['head_image_url'].fillna(value=0, inplace=True)
df['head_image_url'] = df['head_image_url'].map(
    lambda x: 0 if x == ("headImg/20171126/ll15fap1o16y9zfr0ggl3g8xptgo80k9jbnp591d.png") or x == 0 else 1)

df['share_callback'] = np.where(df['share_callback'] < 1, 0, 1)
df['tag'] = np.where(df['tag'].str.match('new'), 1, 0)
df['phone_book'].fillna(value=0, inplace=True)
df['account_num'].fillna(value=0, inplace=True)

df['cert_no'][df['cert_no'].isnull()] = df['card_id'][df['cert_no'].isnull()]
# 有45个身份证号缺失但审核通过的订单， 舍弃不要。
df = df[df['cert_no'].notnull()]

# 取phone前3位
df['phone'][df['phone'].isnull()] = df['phone_user'][df['phone'].isnull()]
df['phone'].fillna(value='0', inplace=True)
df['phone'] = df['phone'].astype(str)
df['phone'][df['phone'].str.len() != 11] = '0'
df['phone'] = df['phone'].str.slice(0, 3)
phones_few = ['143', '106', '478', '162', '812', '165', '163', '196', '179', ]
df['phone'][df['phone'].isin(phones_few)] = '0'
df['phone'] = LabelEncoder().fit_transform(df['phone'])

df[df['phone'].isin(['171', '198'])]['target'].value_counts()
# 处理芝麻信用分 '>600' 更改成600
zmf = [0] * len(df)
xbf = [0] * len(df)
for row, detail in enumerate(df['zmxy_score'].tolist()):
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
                xbf[row] = score
            else:
                zmf[row] = score

df['zmf'] = zmf
df['xbf'] = xbf

df['zmf'][df['zmf'] == 0] = df['zmxyScore'][df['zmf'] == 0]  # 26623
df['xbf'][df['xbf'] == 0] = df['xiaobaiScore'][df['xbf'] == 0]  # 26623
df['zmf'].fillna(value=0, inplace=True)
df['xbf'].fillna(value=0, inplace=True)
zmf_most = df['zmf'][df['zmf'] > 0].value_counts().index[0]
xbf_most = df['xbf'][df['xbf'] > 0].value_counts().index[0]
df['zmf'][df['zmf'] == 0] = zmf_most
df['xbf'][df['xbf'] == 0] = xbf_most

# 芝麻分分类
# bins = pd.IntervalIndex.from_tuples([(0, 600), (600, 700), (700, 800), (800, 1000)])
# df['zmf'] = pd.cut(df['zmf'], bins, labels=False)
# df[['zmf', 'target']].groupby(['zmf'], as_index=False).mean().sort_values(by='target', ascending=False)
# df['zmf'] = LabelEncoder().fit_transform(df['zmf'])

bins = pd.IntervalIndex.from_tuples([(0, 80), (80, 90), (90, 100), (100, 200)])
df['xbf'] = pd.cut(df['xbf'], bins, labels=False)
df[['xbf', 'target']].groupby(['xbf'], as_index=False).mean().sort_values(by='target', ascending=False)
df['xbf'] = LabelEncoder().fit_transform(df['xbf'])


# order_id =9085, 9098的crate_time 是错误的
df = df[df['create_time'] > '2016']
# 把createtime分成月周日小时
# df['create_time'] = pd.to_datetime(df['create_time'])
ft_df = df[['order_id', 'create_time']]
es = ft.EntitySet(id='date')
es = es.entity_from_dataframe(entity_id='date', dataframe=ft_df, index='order_id')
default_trans_primitives = ["day", "month", "weekday", "hour", "year"]
feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity="date", max_depth=1,
                                      trans_primitives=default_trans_primitives)

# feature_matrix['order_id'] = df['order_id']
df = pd.merge(df, feature_matrix, left_on='order_id', right_index=True, how='left')

# 根据身份证号增加性别和年龄 年龄的计算需根据订单创建日期计算
df['age'] = df['YEAR(create_time)'] - df['cert_no'].str.slice(6, 10).astype(int)
df['sex'] = df['cert_no'].str.slice(-2, -1).astype(int) % 2

# 未处理的特征
df.drop(['cert_no_expiry_date', 'regist_useragent', 'cert_no_json', 'bai_qi_shi_detail_json',
         'guanzhu_detail_json', 'mibao_detail_json', 'tongdun_detail_json'],
        axis=1, inplace=True, errors='ignore')
# 已使用的特征
df.drop(['zmxy_score', 'card_id', 'phone_user', 'xiaobaiScore', 'zmxyScore', 'create_time', 'cert_no'], axis=1,
        inplace=True, errors='ignore')
# 与其他特征关联度过高的特征
df.drop(['lease_num', 'install_ment'], axis=1,
        inplace=True, errors='ignore')
missing_values_table(df)
'''
feature = 'WEEKDAY(create_time)'
df[feature].value_counts()
feature_analyse(df, feature, bins=50)
df[feature].dtype
df[df[feature].isnull()].sort_values(by='target').shape
df.shape
df[feature].unique()
df.columns.values
missing_values_table(df)
'''
# merchant 违约率 todo

df.drop(['order_id', 'user_id', 'state'], axis=1, inplace=True, errors='ignore')

df.to_csv(datasets_path + "mibaodata_ml.csv", index=False)
print("mibaodata_ml.csv保存的数据量: {}".format(df.shape))
# In[1]

# 查看各特征关联度
features = ['target',
            'pay_num', 'merchant_store_id', 'merchant_id',
            'added_service', 'bounds_example_id', 'bounds_example_no',
            'goods_type', 'commented', 'accident_insurance',
            'type', 'order_type', 'device_type', 'source', 'distance',
            'disposable_payment_discount', 'disposable_payment_enabled',
             'deposit', 'fingerprint',
            'delivery_way', 'head_image_url', 'recommend_code',
            'regist_channel_type', 'share_callback', 'tag',
            'have_bargain_help', 'face_check', 'face_live_check',
            'company', 'emergency_contact_name', 'phone_book','phone',
            'emergency_contact_phone', 'emergency_contact_relation', 'num',
            'category', 'old_level', 'tongdun_result',
            'guanzhu_result', 'bai_qi_shi_result', 'workplace', 'idcard_pros',
            'occupational_identity_type', 'company_phone', 'device_type_os',
            'regist_device_info', 'ingress_type', 'account_num',
            'zhima_cert_result',
            # 数值类型需转换
            # 'price', 'cost',
            # 实际场景效果不好的特征 # 0.971， 0.930
            ]
corr_df = df[features].astype(float).corr()
corr_df = corr_df.applymap(lambda x: 0 if abs(x)<0.4 else x)
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(corr_df, linewidths=0.1, vmax=1.0,
            square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)
plt.show()
# In[]
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

# ## 评估结果
# accuracy： 97.5%  --- 预测正确的个数占样本总数的比率
# precision： 91.4% --- 预测通过正确的个数占预测通过的比率
# recall：81.4% --- 预测通过正确的个数占实际通过的比率


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
