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
from mlutils import *
import pickle

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
PROJECT_ID = 'mibao'
if os.getcwd().find(PROJECT_ID) == -1:
    os.chdir(PROJECT_ID)
datasets_path = os.getcwd() + '\\datasets\\'
df = pd.read_csv("{}mibao.csv".format(datasets_path), encoding='utf-8', engine='python')
print("初始数据量: {}".format(df.shape))



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

x = df.drop(['target'], axis=1)
y = df['target']

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
x_train = process_data_mibao(x_train)
x_test = process_data_mibao(x_test)

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


lgb_params_binary_logloss = lgb_params.copy()
ret = lgb.cv(lgb_params, train_set, num_boost_round=10000, nfold=5, early_stopping_rounds=100, metrics='binary_logloss',
             seed=42)
lgb_params_binary_logloss['n_estimators'] = len(ret['binary_logloss-mean'])
# Train and make predicions with model
lgb_clf = lgb.LGBMClassifier(**lgb_params_binary_logloss)
lgb_clf.fit(x_train, y_train)
y_pred = lgb_clf.predict(x_test)
add_score(score_df, 'binary_logloss', y_test, y_pred)

lgb_params_auc = lgb_params.copy()
ret = lgb.cv(lgb_params, train_set, num_boost_round=10000, nfold=5, early_stopping_rounds=100, metrics='auc', seed=42)
lgb_params_auc['n_estimators'] = len(ret['auc-mean'])
# Train and make predicions with model
lgb_clf = lgb.LGBMClassifier(**lgb_params_auc)
lgb_clf.fit(x_train, y_train)
y_pred = lgb_clf.predict(x_test)
add_score(score_df, 'auc', y_test, y_pred)
# save model
pickle.dump(lgb_clf, open('mibao_ml.pkl', 'wb'))

feature_importances = lgb_clf.feature_importances_
importance_df = pd.DataFrame({'name': x_train.columns, 'importance': feature_importances})
importance_df.sort_values(by=['importance'], ascending=False, inplace=True)
print(importance_df)
print(score_df)

# In[1]

'''
                 accuracy  precision    recall        f1            confusion_matrix
precision_rs     0.925530   0.899360  0.729770  0.805738   [[4908, 110], [364, 983]]
binary_logloss   0.932600   0.777283  0.890306  0.829964  [[4889, 300], [129, 1047]]
auc              0.935114   0.811433  0.873003  0.841093  [[4859, 254], [159, 1093]]
neg_log_loss_rs  0.938570   0.887358  0.812918  0.848508  [[4879, 139], [252, 1095]]
recall_rs        0.938570   0.883013  0.818114  0.849326  [[4872, 146], [245, 1102]]
accuracy_rs      0.939042   0.888259  0.814402  0.849729  [[4880, 138], [250, 1097]]
roc_auc_rs       0.939670   0.888620  0.817372  0.851508  [[4880, 138], [246, 1101]]
f1_rs            0.941084   0.889423  0.824053  0.855491  [[4880, 138], [237, 1110]]

                accuracy  precision    recall        f1           confusion_matrix
binary_logloss  0.975000   0.919259  0.961270  0.939796  [[4960, 109], [50, 1241]]
auc             0.975157   0.918519  0.962733  0.940106  [[4962, 110], [48, 1240]]
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
score_df.sort_values(by='accuracy', inplace=True)
score_df

# In[2]
exit('ml')
'''
                      accuracy  precision    recall        f1           confusion_matrix
auc                   0.977690   0.933185  0.960275  0.946536   [[4966, 90], [52, 1257]]
binary_logloss        0.972192   0.914625  0.952087  0.932980  [[4956, 115], [62, 1232]]
accuracy_rs           0.978162   0.963190  0.932442  0.947567   [[4970, 48], [91, 1256]]
average_precision_rs  0.978476   0.961128  0.936154  0.948477   [[4967, 51], [86, 1261]]
f1_rs                 0.977062   0.960153  0.930215  0.944947   [[4966, 52], [94, 1253]]
f1_micro_rs           0.977848   0.962423  0.931700  0.946813   [[4969, 49], [92, 1255]]
f1_macro_rs           0.978947   0.963331  0.936154  0.949548   [[4970, 48], [86, 1261]]
f1_weighted_rs        0.977376   0.961627  0.930215  0.945660   [[4968, 50], [94, 1253]]
neg_log_loss_rs       0.978162   0.963190  0.932442  0.947567   [[4970, 48], [91, 1256]]
precision_rs          0.979104   0.962652  0.937639  0.949981   [[4969, 49], [84, 1263]]
recall_rs             0.978476   0.961832  0.935412  0.948438   [[4968, 50], [87, 1260]]
roc_auc_rs            0.978162   0.961069  0.934670  0.947685   [[4967, 51], [88, 1259]]
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