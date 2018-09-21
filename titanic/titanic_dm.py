# coding: utf-8

import csv
import json
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import os
from sklearn.preprocessing import LabelEncoder
# Suppress warnings
import warnings



warnings.filterwarnings('ignore')
# to make output display better
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 2000)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['font.sans-serif'] = ['Simhei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# read large csv file
csv.field_size_limit(100000000)

# ## 获取数据
# 数据已经从数据库中导出成csv文件，直接读取即可。后面数据的读取更改为从备份数据库直接读取，不仅可以保证数据的完整，还可以避免重名字段处理的麻烦。
PROJECT_ROOT_DIR = os.getcwd()
DATA_ID = "train.csv"
DATASETS_PATH = os.path.join(PROJECT_ROOT_DIR, "titanic\datasets", DATA_ID)
df_alldata = pd.read_csv(DATASETS_PATH, encoding='utf-8', engine='python')
print("初始数据量: {}".format(df_alldata.shape))

# ## 数据简单计量分析
# 查看头尾数据
df_alldata

# 所有特征值
df_alldata.columns.values

# 特征选择
# f_classif
# mutual_info_classif

# 我们并不需要所有的特征值，筛选出一些可能有用的特质值
df = df_alldata.dropna(axis=1, how='all')
print("筛选出所有可能有用特征后的数据量: {}".format(df.shape))

# 查看数据类型
df.dtypes.value_counts()
# 缺失值比率
missing_values_table(df)
# 特征中不同值得个数
df.select_dtypes('object').apply(pd.Series.nunique, axis=0)
#  数值描述
df.describe()
# 类别描述
df.describe(include='O')

# 开始清理数据
features = ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin',
            'Embarked']
label = 'Survived'
feature = 'Pclass'
plt.figure()
plt.hist(df[feature])
feature_analyse(df, feature, label)
feature_kdeplot(df, feature, label)

features = ['check_result', 'result', 'pay_num', 'channel', 'goods_type', 'type', 'order_type',
            'source', 'phone_book', 'old_level', 'sex', 'create_hour', 'age_band', 'zmf_score_band',
            'xbf_score_band', ]
df = df[features]
# 类别特征全部转换成数字
for feature in features:
    df[feature] = LabelEncoder().fit_transform(df[feature])

print("保存的数据量: {}".format(df.shape))
df.to_csv(os.path.join(PROJECT_ROOT_DIR, "datasets", "mibaodata_ml.csv"), index=False)

# 查看各特征关联度
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df.astype(float).corr(), linewidths=0.1, vmax=1.0,
            square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)

# ## 机器学习训练、预测

import time
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

# to make output display better
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 1000)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
# read large csv file
PROJECT_ROOT_DIR = os.getcwd()
DATA_ID = "mibaodata_ml.csv"
DATASETS_PATH = os.path.join(PROJECT_ROOT_DIR, "datasets", DATA_ID)
# Get Data
df = pd.read_csv(DATASETS_PATH, encoding='utf-8', engine='python')
print("ML初始数据量: {}".format(df.shape))

x = df.drop(['check_result'], axis=1)
y = df['check_result']
## Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# 保存所有模型得分
def add_score(score_df, name, runtime, y_pred, y_test):
    score_df[name] = [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred),
                      f1_score(y_test, y_pred), runtime, confusion_matrix(y_test, y_pred)]

    return score_df


score_df = pd.DataFrame(index=['accuracy', 'precision', 'recall', 'f1', 'runtime', 'confusion_matrix'])

rnd_clf = RandomForestClassifier(random_state=0)
starttime = time.clock()
rnd_clf.fit(x_train, y_train)
y_pred = rnd_clf.predict(x_test)
add_score(score_df, rnd_clf.__class__.__name__, time.clock() - starttime, y_pred, y_test)
y_train_pred = rnd_clf.predict(x_train)
add_score(score_df, rnd_clf.__class__.__name__ + '_Train', time.clock() - starttime, y_train_pred, y_train)

xgb_clf = XGBClassifier(random_state=0)
starttime = time.clock()
xgb_clf.fit(x_train, y_train)
y_pred = xgb_clf.predict(x_test)
add_score(score_df, xgb_clf.__class__.__name__, time.clock() - starttime, y_pred, y_test)
y_train_pred = xgb_clf.predict(x_train)
add_score(score_df, xgb_clf.__class__.__name__ + '_Train', time.clock() - starttime, y_train_pred, y_train)

print(score_df)

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
