# coding: utf-8
import time
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_predict, train_test_split, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold
import lightgbm as lgb
import random

import pickle
from sklearn.externals import joblib
import json
from mltools import *
from explore_data_utils import *


# to make output display better
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 70)
pd.set_option('display.width', 1000)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
# read large csv file
time_started = time.clock()
# 设置随机种子
# np.random.seed(88)
# ## 获取数据
all_ml_data_df = pd.read_csv(os.path.join(workdir, "mibaodata_ml.csv"), encoding='utf-8', engine='python')
print("初始数据量: {}".format(all_ml_data_df.shape))
df = all_ml_data_df.copy()


system_credit_check_unpass_canceled_df = df[df['state'] == 'system_credit_check_unpass_canceled' ]
user_canceled_system_credit_unpass_df = df[df['state'] == 'user_canceled_system_credit_unpass' ]
df = df[df['state'] != 'user_canceled_system_credit_unpass' ]
df = df[df['state'] != 'system_credit_check_unpass_canceled' ]
print("训练数据量: {}".format(df.shape))
result_df = df[['order_id','target']]


x = df.drop(['target', 'state'], axis=1, errors='ignore')
y = df['target']
# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

x_c = system_credit_check_unpass_canceled_df.drop(['target', 'state'], axis=1, errors='ignore')
y_c = system_credit_check_unpass_canceled_df['target']
x_c2 = user_canceled_system_credit_unpass_df.drop(['target', 'state'], axis=1, errors='ignore')
y_c2 = user_canceled_system_credit_unpass_df['target']
# x_train = pd.concat([x_train, x_c])
# y_train = pd.concat([y_train, y_c])
# x_train = pd.concat([x_train, x_c2])
# y_train = pd.concat([y_train, y_c2])


score_df = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'])

# Create a training and testing dataset
train_set = lgb.Dataset(data=x_train, label=y_train)
test_set = lgb.Dataset(data=x_test, label=y_test)
# Get default hyperparameters
lgb_clf = lgb.LGBMClassifier()
lgb_params = lgb_clf.get_params()


# LightBGM with Random Search
param_grid = {
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'n_estimators': range(1, 500),
    'num_leaves': list(range(20, 150)),
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)),
    #
    # 'subsample_for_bin': list(range(20000, 300000, 20000)),
    # 'min_child_samples': list(range(20, 500, 5)),
    # 'reg_alpha': list(np.linspace(0, 1)),
    # 'reg_lambda': list(np.linspace(0, 1)),
    # 'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    # 'subsample': list(np.linspace(0.5, 1, 100)),
    # 'is_unbalance': [True, False]
}
scorings = ['accuracy', 'precision', 'recall', 'roc_auc', 'f1', 'average_precision', 'f1_micro', 'f1_macro',
            'f1_weighted', 'neg_log_loss', ]
scorings = ['neg_log_loss', 'accuracy', 'precision', 'recall', 'roc_auc', 'f1', ]

for scoring in scorings:
    lgb_clf = lgb.LGBMClassifier()
    rnd_search = RandomizedSearchCV(lgb_clf, param_distributions=param_grid, n_iter=5, cv=5, scoring=scoring, n_jobs=-1)
    rnd_search.fit(x_train, y_train)
    # rnd_search.best_params_
    # rnd_search.best_estimator_
    # rnd_search.best_score_
    # cvres = rnd_search.cv_results_

    # Train and make predicions with model
    lgb_clf = rnd_search.best_estimator_
    lgb_clf.fit(x_train, y_train)
    y_pred = lgb_clf.predict(x_test)
    add_score(score_df, scoring + '_rs', y_test, y_pred)
    print(score_df)

# feature_importances = rnd_search.best_estimator_.feature_importances_
# importance_df = pd.DataFrame({'name': x_train.columns, 'importance': feature_importances})
# importance_df.sort_values(by=['importance'], ascending=False, inplace=True)
# print(importance_df)
print('run time: {:.2f}'.format(time.clock() - time_started))
score_df.sort_values(by='recall', inplace=True)
score_df

# In[2]
exit('ml')
'''

'''

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
6. 下单使用的设备，通过什么客户端下单（微信、支付宝、京东、网页）
7. 是否有推荐人， 推荐人是否通过审核
#. 租借个数

 b. 创造未被保存到数据库中的特征：年化利率,是否有2个手机号。
'''

