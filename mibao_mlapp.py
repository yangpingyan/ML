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
np.random.seed(88)
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


print(score_df)
# result_df['predict'] = y_pred
# result_df.to_csv(os.path.join(workdir,"mibao_mlresult.csv"), index=False)


# In[1]


'''
is_unbalance:
                accuracy  precision    recall        f1              confusion_matrix
binary_logloss  0.911778   0.598563  0.891795  0.716332      [[5389, 503], [91, 750]]
auc             0.924105   0.643728  0.878716  0.743087     [[5483, 409], [102, 739]]
auc_alldata     0.938507   0.686877  0.937907  0.793000  [[55255, 3615], [525, 7930]]
                accuracy  precision    recall        f1              confusion_matrix
binary_logloss  0.949502   0.856330  0.715815  0.779793     [[5791, 101], [239, 602]]
auc             0.950097   0.859175  0.718193  0.782383      [[5793, 99], [237, 604]]
auc_alldata     0.963520   0.922049  0.775044  0.842180  [[58316, 554], [1902, 6553]]


                           name  importance
47                        price         408
39                          age         398
43                  final_score         387
48                         cost         351
23                        phone         347
41                          zmf         295
46                         hour         265
42                          xbf         248
0                   merchant_id         194
8                   device_type         179
45                      weekday         168
6            accident_insurance         148
3                    goods_type         146
26                     category         129
9                        source         126
5                     commented         119
51            bounds_example_id         112
49                   phone_book         109
16               head_image_url          93
29               guanzhu_result          91
18          regist_channel_type          79
38            zhima_cert_result          76
32                  idcard_pros          60
20                          tag          56
13            merchant_store_id          49
1                       pay_num          48
4                    lease_term          47
30            bai_qi_shi_result          46
34               device_type_os          46
28               tongdun_result          42
27                    old_level          41
40                          sex          37
33   occupational_identity_type          35
22                   face_check          29
36                 ingress_type          25
11  disposable_payment_discount          21
24                      company          18
15                 delivery_way          17
21            have_bargain_help          17
7                    order_type          16
44               final_decision          16
14                  fingerprint          16
25                company_phone          16
50              face_live_check          13
35           regist_device_info          12
17               recommend_code           8
2                 added_service           7
19               share_callback           5
10                     distance           5
37               baiqishi_score           4
31                    workplace           0
12   disposable_payment_enabled           0



'''


# LightBGM with Random Search
param_grid = {
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'n_estimators': range(100, 600),
    'num_leaves': list(range(20, 200)),
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)),
    #
    # 'subsample_for_bin': list(range(20000, 300000, 20000)),
    # 'min_child_samples': list(range(20, 500, 5)),
    # 'reg_alpha': list(np.linspace(0, 1)),
    # 'reg_lambda': list(np.linspace(0, 1)),
    # 'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    # 'subsample': list(np.linspace(0.5, 1, 100)),
    'is_unbalance': [True]
}

scorings = ['f1_micro','neg_log_loss', 'accuracy', 'precision', 'recall', 'roc_auc', 'f1', 'average_precision',
            'f1_macro', 'f1_weighted']
scorings = [ 'precision', 'roc_auc', 'f1']
# scorings = ['f1_micro']
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

    with open('{}.json'.format(scoring), 'w') as f:
        json.dump(lgb_clf.get_params(), f, indent=4)
    lgb_clf.fit(x_train, y_train)
    y_pred = lgb_clf.predict(x_test)
    add_score(score_df, scoring + '_rs', y_test, y_pred)
    print(score_df)

# feature_importances = rnd_search.best_estimator_.feature_importances_
# importance_df = pd.DataFrame({'name': x_train.columns, 'importance': feature_importances})
# importance_df.sort_values(by=['importance'], ascending=False, inplace=True)
# print(importance_df)
print('run time: {:.2f}'.format(time.clock() - time_started))
score_df.sort_values(by='f1', inplace=True)
print(score_df)

# In[2]
# exit('ml')
'''
                      accuracy  precision    recall        f1              confusion_matrix
binary_logloss        0.911778   0.598563  0.891795  0.716332      [[5389, 503], [91, 750]]
auc                   0.924105   0.643728  0.878716  0.743087     [[5483, 409], [102, 739]]
precision_rs          0.948463   0.870871  0.689655  0.769741      [[5806, 86], [261, 580]]
f1_macro_rs           0.947869   0.855072  0.701546  0.770738     [[5792, 100], [251, 590]]
neg_log_loss_rs       0.948611   0.863436  0.699168  0.772668      [[5799, 93], [253, 588]]
roc_auc_rs            0.950097   0.863309  0.713436  0.781250      [[5797, 95], [241, 600]]
accuracy_rs           0.950542   0.872434  0.707491  0.781353      [[5805, 87], [246, 595]]
f1_weighted_rs        0.950691   0.865136  0.717004  0.784135      [[5798, 94], [238, 603]]
f1_rs                 0.951285   0.874453  0.712247  0.785059      [[5806, 86], [242, 599]]
average_precision_rs  0.950839   0.864286  0.719382  0.785204      [[5797, 95], [236, 605]]
recall_rs             0.950988   0.865522  0.719382  0.785714      [[5798, 94], [236, 605]]
f1_micro_rs           0.951285   0.868006  0.719382  0.786736      [[5800, 92], [236, 605]]
auc_alldata           0.938507   0.686877  0.937907  0.793000  [[55255, 3615], [525, 7930]]

binary_logloss   0.911778   0.598563  0.891795  0.716332   [[5389, 503], [91, 750]]
auc              0.924105   0.643728  0.878716  0.743087  [[5483, 409], [102, 739]]
f1_micro_rs      0.942225   0.747807  0.810939  0.778095  [[5662, 230], [159, 682]]
neg_log_loss_rs  0.944601   0.768349  0.796671  0.782253  [[5690, 202], [171, 670]]
accuracy_rs      0.947572   0.811224  0.756243  0.782769  [[5744, 148], [205, 636]]
precision_rs     0.949354   0.814070  0.770511  0.791692  [[5744, 148], [193, 648]]


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
9. 利率
10.红包

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


'''
classifiers = [
    lgb.LGBMClassifier(),  # 0.931343， 0.833524
    # RandomForestClassifier(), #0.924745, 0.811788
    # KNeighborsClassifier(3),
    # # SVC(probability=True),
    # DecisionTreeClassifier(),
    # AdaBoostClassifier(),
    # GradientBoostingClassifier(),
    # GaussianNB(),
    # LinearDiscriminantAnalysis(),
    # QuadraticDiscriminantAnalysis(),
    # LogisticRegression(),
    # SGDClassifier(max_iter=5),
    # Perceptron(),
    # XGBClassifier()
]
'''