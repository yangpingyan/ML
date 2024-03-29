# coding: utf-8
#!/usr/bin/env python
# coding: utf-8
# @Time : 2018/9/28 16:42
# @Author : yangpingyan@gmail.com

import csv
import json
import pandas as pd
import numpy as np
import os

from explore_data_utils import *
from mltools import *
from mldata import *
import operator
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# to make output display better
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 2000)

# read large csv file
csv.field_size_limit(100000000)

all_data_merged_df = get_order_data()
# In[]
df = all_data_merged_df.copy()
df.shape
df['state'].value_counts()
df['state_cao'].value_counts()
# 若state字段有新的状态产生， 抛出异常
state_values_newest = df['state'].unique().tolist()
print(list(set(state_values_newest).difference(set(state_values))))
assert (len(list(set(state_values_newest).difference(set(state_values)))) == 0)

# 标注人工审核结果于target字段
df['target'] = None
df.loc[df['state_cao'].isin(['manual_check_fail', 'self_check_fail']), 'target'] = 0
df.loc[df['state_cao'].isin(['manual_check_success', 'merchant_self_check_success']), 'target'] = 1
df.loc[df['state'].isin(pass_state_values), 'target'] = 1
df.loc[df['state'].isin(failure_state_values), 'target'] = 0
df = df[df['target'].notnull()]
df['target'].value_counts()
df.shape
# 丢弃不需要的数据
# 丢弃白名单用户
risk_white_list_df = read_mlfile('risk_white_list', ['user_id'])
user_ids = risk_white_list_df['user_id'].values
df = df[df['user_id'].isin(user_ids) != True]
# 丢弃joke为1的order
df = df[df['joke'] != 1]
# 去除测试数据和内部员工数据
df = df[df['cancel_reason'].str.contains('测试') != True]
df = df[df['check_remark'].str.contains('测试') != True]
# 去除命中商户白名单的订单
df = df[df['hit_merchant_white_list'].str.contains('01') != True]

# 丢弃不需要的特征
df.drop(
    ['tongdun_detail_json', 'mibao_result', 'order_number', 'cancel_reason', 'hit_merchant_white_list', 'check_remark',
     'joke', 'mibao_remark', 'tongdun_remark', 'bai_qi_shi_remark', 'guanzhu_remark'],
    axis=1,
    inplace=True, errors='ignore')

# print(set(df.columns.tolist()) - set(df_sql.columns.tolist()))
# 保存数据
# df.to_csv("mibaodata_merged.csv", index=False)
# print("mibaodata_merged.csv saved with shape {}".format(df.shape))
# missing_values_table(df)


# 数据清洗
df = process_data_mibao(df)
df.to_csv(os.path.join(workdir, "mibaodata_ml.csv"), index=False)
print("mibaodata_ml.csv保存的数据量: {}".format(df.shape))
# In[]

exit('mergedata')

import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['font.sans-serif'] = ['Simhei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 总订单 98078， 经过审核的订单有 73560, 未经审核被取消的

# In[1]
exit('dm')

# 查看各特征关联度
features = ['target',
            'pay_num', 'merchant_store_id', 'merchant_id',
            'added_service', 'bounds_example_id',
            'goods_type', 'commented', 'accident_insurance',
            'type', 'order_type', 'device_type', 'source', 'distance',
            'disposable_payment_discount', 'disposable_payment_enabled',
            'deposit', 'fingerprint',
            'delivery_way', 'head_image_url', 'recommend_code',
            'regist_channel_type', 'share_callback', 'tag',
            'have_bargain_help', 'face_check', 'face_live_check',
            'company', 'phone_book', 'phone',
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
corr_df = corr_df.applymap(lambda x: 0 if abs(x) < 0.4 else x)
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(corr_df, linewidths=0.1, vmax=1.0,
            square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)
plt.show()
# In[]

'''
现有数据特征挖掘（与客户和订单信息相关的数据都可以拿来做特征）：
1. 同盾和白骑士中的内容详情
2. IP地址、收货地址、身份证地址、居住地址中的关联
3. 优惠券，额外服务意外险
4. 首付金额、每日租金
6. 下单使用的设备，通过什么客户端下单（微信、支付宝、京东、网页）
7. 是否有推荐人， 推荐人是否通过审核
#. 租借个数
# 读取并处理表 用户登陆使用情况， 在mango数据库
 b. 创造未被保存到数据库中的特征：年化利率,是否有2个手机号。
'''


# ## 评估结果
# accuracy： 97.5%  --- 预测正确的个数占样本总数的比率
# precision： 91.4% --- 预测通过正确的个数占预测通过的比率
# recall：81.4% --- 预测通过正确的个数占实际通过的比率


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


# 芝麻分分类
# bins = pd.IntervalIndex.from_tuples([(0, 600), (600, 700), (700, 800), (800, 1000)])
# df['zmf'] = pd.cut(df['zmf'], bins, labels=False)
# df[['zmf', 'target']].groupby(['zmf'], as_index=False).mean().sort_values(by='target', ascending=False)
# df['zmf'] = LabelEncoder().fit_transform(df['zmf'])
# 小白分分类
# bins = pd.IntervalIndex.from_tuples([(0, 80), (80, 90), (90, 100), (100, 200)])
# df['xbf'] = pd.cut(df['xbf'], bins, labels=False)
# df[['xbf', 'target']].groupby(['xbf'], as_index=False).mean().sort_values(by='target', ascending=False)
# df['xbf'] = LabelEncoder().fit_transform(df['xbf'])
