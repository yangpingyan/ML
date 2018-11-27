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
from mldata import *

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
all_data_ml_df = pd.read_csv(os.path.join(workdir, "mibaodata_ml.csv"), encoding='utf-8', engine='python')
print("初始数据量: {}".format(all_data_ml_df.shape))
df = all_data_ml_df.copy()

result_df = df[['order_id', 'target']]

print(list(set(df.columns.tolist()).difference(set(mibao_ml_features))))

x = df[mibao_ml_features]
y = df['target']
# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

score_df = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'])

# Create a training and testing dataset
train_set = lgb.Dataset(data=x_train, label=y_train)
test_set = lgb.Dataset(data=x_test, label=y_test)
# Get default hyperparameters
lgb_clf = lgb.LGBMClassifier()
lgb_params = lgb_clf.get_params()
# Number of estimators will be found using early stopping
if 'n_estimators' in lgb_params.keys():
    del lgb_params['n_estimators']
    # Perform n_folds cross validation
# param['metric'] = ['auc', 'binary_logloss']
lgb_params['is_unbalance'] = True
# lgb_params['boosting_type'] = 'dart'

lgb_params_binary_logloss = lgb_params.copy()
ret = lgb.cv(lgb_params_binary_logloss, train_set, num_boost_round=10000, nfold=5, early_stopping_rounds=100, metrics='binary_logloss')
lgb_params_binary_logloss['n_estimators'] = len(ret['binary_logloss-mean'])
# Train and make predicions with model
lgb_clf = lgb.LGBMClassifier(**lgb_params_binary_logloss)
lgb_clf.fit(x_train, y_train)
y_pred = lgb_clf.predict(x_test)
add_score(score_df, 'binary_logloss', y_test, y_pred)

lgb_params_auc = lgb_params.copy()
ret = lgb.cv(lgb_params_auc, train_set, num_boost_round=10000, nfold=5, early_stopping_rounds=100, metrics='auc')
lgb_params_auc['n_estimators'] = len(ret['auc-mean'])
# Train and make predicions with model
lgb_clf = lgb.LGBMClassifier(**lgb_params_auc)
lgb_clf.fit(x_train, y_train)
y_pred = lgb_clf.predict(x_test)
add_score(score_df, 'auc', y_test, y_pred)
# save model
# pickle.dump(lgb_clf, open('mibao_ml.pkl', 'wb'))
with open('lgb_params.json', 'w') as f:
    json.dump(lgb_params_auc, f, indent=4)

feature_importances = lgb_clf.feature_importances_
importance_df = pd.DataFrame({'name': x_train.columns, 'importance': feature_importances})
importance_df.sort_values(by=['importance'], ascending=False, inplace=True)
print(importance_df)

y_pred = lgb_clf.predict(x)
add_score(score_df, 'auc_alldata', y_test=y, y_pred=y_pred)
print(score_df)
# result_df['predict'] = y_pred
# result_df.to_csv(os.path.join(workdir,"mibao_mlresult.csv"), index=False)


# In[1]

print("Mission Complete")
exit("mibao_ml")
'''
is_unbalance:
                accuracy  precision    recall        f1            confusion_matrix
binary_logloss  0.980604   0.874033  0.981390  0.924605    [[5731, 114], [15, 791]]
auc             0.980604   0.873208  0.982630  0.924694    [[5730, 115], [14, 792]]
auc_alldata     0.983790   0.890222  0.987684  0.936424  [[57485, 979], [99, 7939]]

                accuracy  precision    recall        f1            confusion_matrix
binary_logloss  0.982108   0.885778  0.978960  0.930041    [[5741, 102], [17, 791]]
auc             0.982559   0.889640  0.977723  0.931604     [[5745, 98], [18, 790]]
auc_alldata     0.984948   0.896719  0.989425  0.940794  [[57548, 916], [85, 7953]]

                           name  importance
47                  final_score         139
9                          type         122
51                        price          99
52                         cost          93
3             bounds_example_id          88
12                       source          79
11                  device_type          73
45                          zmf          63
0                   merchant_id          58
43                          age          52
4             bounds_example_no          51
27                        phone          51
30                     category          47
8            accident_insurance          44
7                     commented          42
50                         hour          42
16            merchant_store_id          41
5                    goods_type          39
46                          xbf          36
33               guanzhu_result          36
20               head_image_url          32
49                      weekday          26
32               tongdun_result          25
1                       pay_num          23
24                          tag          18
22          regist_channel_type          18
18                  fingerprint          16
31                    old_level          13
42            zhima_cert_result          12
23               share_callback          10
38               device_type_os           8
48               final_decision           5
34            bai_qi_shi_result           5
40                 ingress_type           4
37   occupational_identity_type           3
6                    lease_term           3
26                   face_check           3
39           regist_device_info           2
21               recommend_code           2
2                 added_service           2
36                  idcard_pros           1
29                company_phone           1
19                 delivery_way           1
44                          sex           1
25            have_bargain_help           1
35                    workplace           0
41               baiqishi_score           0
17                      deposit           0
15   disposable_payment_enabled           0
14  disposable_payment_discount           0
10                   order_type           0
13                     distance           0
28                      company           0


'''

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

scorings = ['neg_log_loss', 'accuracy', 'precision', 'recall', 'roc_auc', 'f1', 'f1_micro', 'average_precision',
            'f1_macro', 'f1_weighted']

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
print(score_df)

# In[2]
exit('ml')
'''
                      accuracy  precision    recall        f1            confusion_matrix
precision_rs          0.975833   0.951327  0.834411  0.889042    [[5856, 33], [128, 645]]
roc_auc_rs            0.981837   0.935829  0.905563  0.920447     [[5841, 48], [73, 700]]
neg_log_loss_rs       0.982288   0.938420  0.906856  0.922368     [[5843, 46], [72, 701]]
recall_rs             0.981987   0.935915  0.906856  0.921156     [[5841, 48], [72, 701]]
f1_weighted_rs        0.982137   0.937166  0.906856  0.921762     [[5842, 47], [72, 701]]
f1_macro_rs           0.982438   0.938503  0.908150  0.923077     [[5843, 46], [71, 702]]
f1_micro_rs           0.982588   0.938585  0.909444  0.923784     [[5843, 46], [70, 703]]
average_precision_rs  0.982288   0.936085  0.909444  0.922572     [[5841, 48], [70, 703]]
f1_rs                 0.982137   0.933687  0.910737  0.922069     [[5839, 50], [69, 704]]
accuracy_rs           0.982738   0.937500  0.912031  0.924590     [[5842, 47], [68, 705]]
binary_logloss        0.974632   0.851163  0.946960  0.896509    [[5761, 128], [41, 732]]
auc                   0.979135   0.881010  0.948254  0.913396     [[5790, 99], [40, 733]]
auc_alldata           0.987151   0.913343  0.989110  0.949718  [[57679, 767], [89, 8084]]
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

# ## 评估结果
# accuracy： 97.5%  --- 预测正确的个数占样本总数的比率
# precision： 91.4% --- 预测通过正确的个数占预测通过的比率
# recall：81.4% --- 预测通过正确的个数占实际通过的比率

# In[42]:

'''
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

'''

'''
feature = 'MONTH(create_time)'
df[feature].value_counts()
feature_analyse(df, feature, bins=50)
df[feature].dtype
df[df[feature].isnull()].sort_values(by='target').shape
df.shape
df[feature].unique()
df.columns.values
missing_values_table(df)
'''
