# coding: utf-8

# # 蜜宝大数据风控解决方案
#
# ---
#
# ## 会议主题：
# 1. 审核大数据风控可行性。
# 2. 工作计划。
#
#
# ## 数据挖掘工作流程
#
# 目前大数据风控做的第一件事是数据挖掘工作，数据挖掘的工作流程分下面七步完成：
#
# 1. 目标或问题定义。
# 2. 获取数据。
# 3. 数据分析。
# 4. 数据清洗、特征处理。
# 5. 机器学习训练、预测。
# 6. 结果评估和报告。
# 7. 总结。
#
#
# ## 目标或问题定义
#
# 当我们面对客户提交的租赁设备订单请求时，我们有2个核心问题需要解决，一个是这个客户信用如何，是不是来欺诈的；另一个是这个客户是信用良好客户，但我们不确定这个设备的价格是否超出他所能承受的范围。因此，我们的任务目标是两个：
# 1. 客户分类。把客户分成审核通过和审核拒绝两类。
# 2. 确定客户信用额度。
# 接下来的数据挖掘工作是实现客户分类的。

# ## 开始数据挖掘工作
# 先做些代码初始化

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


def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# 特征分析
def feature_analyse(df, col, bins=10):
    if df[col].dtype != 'O':
        col_band = col + '_band'
        df[col_band] = pd.cut(df[col], bins).astype(str)
        col_ana = col_band
    else:
        col_ana = col

    print(df[col_ana].describe())
    print("-------------------------------------------")
    pass_df = pd.DataFrame({'pass': df[df['check_result'] == 1][col_ana].value_counts()})
    reject_df = pd.DataFrame({'reject': df[df['check_result'] == 0][col_ana].value_counts()})
    all_df = pd.DataFrame({'all': df[col_ana].value_counts()})
    analyse_df = all_df.merge(pass_df, how='outer', left_index=True, right_index=True)
    analyse_df = analyse_df.merge(reject_df, how='outer', left_index=True, right_index=True)
    analyse_df['pass_rate'] = analyse_df['pass'] / analyse_df['all']
    analyse_df.sort_values(by='pass_rate', inplace=True, ascending=False)
    analyse_df.fillna(value=0, inplace=True)
    print(analyse_df)
    plt.bar(analyse_df.index, analyse_df['pass_rate'])
    plt.ylabel('Pass Rate')


# KDE plot
def feature_kdeplot(df, feature):
    sns.kdeplot(df.loc[df['check_result'] == 0, feature], label='reject')
    sns.kdeplot(df.loc[df['check_result'] == 1, feature], label='pass')

    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.title('Distribution of ' + feature)


def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better.

    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance

    Returns:
        shows a plot of the 15 most importance features

        df (dataframe): feature importances sorted by importance (highest to lowest)
        with a column for normalized importance
        """

    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()

    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()

    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))),
            df['importance_normalized'].head(15),
            align='center', edgecolor='k')

    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))

    # Plot labeling
    plt.xlabel('Normalized Importance');
    plt.title('Feature Importances')
    plt.show()

    return df


warnings.filterwarnings('ignore')
# to make output display better
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 1000)
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
DATA_ID = "学校数据.csv"
DATASETS_PATH = os.path.join(PROJECT_ROOT_DIR, "datasets", DATA_ID)
df_alldata = pd.read_csv(DATASETS_PATH, encoding='utf-8', engine='python')
print("初始数据量: {}".format(df_alldata.shape))

# ## 数据简单计量分析
# 首5行数据
df_alldata.head()
# 最后5行数据
df_alldata.tail()
# 所有特征值
df_alldata.columns.values

# 我们并不需要所有的特征值，筛选出一些可能有用的特质值
df = df_alldata.dropna(axis=1, how='all')

features = ['create_time', 'goods_name', 'cost', 'discount', 'pay_num', 'added_service', 'first_pay', 'channel',
            'pay_type', 'merchant_id', 'goods_type', 'lease_term', 'daily_rent', 'accident_insurance', 'type',
            'freeze_money', 'ip', 'releted', 'order_type', 'source', 'disposable_payment_discount',
            'disposable_payment_enabled', 'lease_num', 'original_daily_rent', 'deposit', 'zmxy_score', 'card_id',
            'contact', 'phone', 'provice', 'city', 'regoin', 'receive_address', 'emergency_contact_name', 'phone_book',
            'emergency_contact_phone', 'emergency_contact_relation', 'type.1', 'detail_json', 'price', 'old_level']
result = ['state', 'cancel_reason', 'check_result', 'check_remark', 'result']
df = df[result + features]
print("筛选出所有可能有用特征后的数据量: {}".format(df.shape))
# 数据的起止时间段
print("数据起止时间段：{} -- {}".format(df['create_time'].iloc[0], df['create_time'].iloc[-1]))
# 订单审核结果分类
df['check_result'].value_counts()
# 订单状态
df['state'].value_counts()

# 查看非空值个数， 数据类型
df.info()

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
print("初始数据量: {}".format(df.shape))

# 丢弃身份证号为空的数据
df.dropna(subset=['card_id'], inplace=True)
print("去除无身份证号后的数据量: {}".format(df.shape))


# 取有审核结果的数据
df = df[df['check_result'].str.contains('SUCCESS|FAILURE', na=False)]
print("去除未经机审用户后的数据量: {}".format(df.shape))


# 去除测试数据和内部员工数据
df = df[df['cancel_reason'].str.contains('测试|内部员工') != True]
df = df[df['check_remark'].str.contains('测试|内部员工') != True]
print("去除测试数据和内部员工后的数据量: {}".format(df.shape))
# 去掉用户自己取消的数据   问题：即使用户取消了，仍然会有审核？？
df = df[df['state'].str.match('user_canceled') != True]
print("去除用户自己取消后的数据量: {}".format(df.shape))
# 去除身份证重复的订单：
df.drop_duplicates(subset=['card_id'], keep='last', inplace=True)
print("去除身份证重复的订单后的数据量: {}".format(df.shape))

# 所有字符串变成大写字母
objs_df = pd.DataFrame({"isobj": pd.Series(df.dtypes == 'object')})
df[objs_df[objs_df['isobj'] == True].index.values] = df[objs_df[objs_df['isobj'] == True].index.values].applymap(
    lambda x: x.upper() if isinstance(x, str) else x)

# 隐藏身份证信息
df['card_id'] = df['card_id'].map(lambda x: x.replace(x[10:16], '******') if isinstance(x, str) else x)

# 处理running_overdue 和 return_overdue 的逾期 的 check_result
df.loc[df['state'].str.contains('overdue') == True, 'check_result'] = 'FAILURE'
df['check_result'] = df['check_result'].apply(lambda x: 1 if 'SUCCESS' in x else 0)

# 有phone_book的赋值成1， 空的赋值成0
df['phone_book'][df['phone_book'].notnull()] = 1
df['phone_book'][df['phone_book'].isnull()] = 0
# 根据create_time 按时间段分类
df['create_hour'] = df['create_time'].map(lambda x: int(x[-8:-6]))
df['create_time_cat'] = df['create_hour'].map(lambda x: 0 if 0 < x < 7 else 1)
# 同盾白骑士审核结果统一
df['result'] = df['result'].map(lambda x: x.upper() if isinstance(x, str) else 'NODATA')
df['result'][df['result'].str.match('ACCEPT')] = 'PASS'
# 有emergency_contact_phone的赋值成1， 空的赋值成0
df['emergency_contact_phone'][df['emergency_contact_phone'].notnull()] = 1
df['emergency_contact_phone'][df['emergency_contact_phone'].isnull()] = 0

# 处理芝麻信用分 '>600' 更改成600
row = 0
zmf = [0] * len(df)
xbf = [0] * len(df)
for x in df['zmxy_score']:
    # print(x, row)
    if isinstance(x, str):
        if '/' in x:
            score = x.split('/')
            xbf[row] = 0 if score[0] == '' else (float(score[0]))
            zmf[row] = 0 if score[1] == '' else (float(score[1]))
            # print(score, row)
        elif '>' in x:
            zmf[row] = 600
        else:
            score = float(x)
            if score <= 200:
                xbf[row] = (score)
            else:
                zmf[row] = (score)

    row += 1

df['zmf_score'] = zmf
df['xbf_score'] = xbf
df['zmf_score'][df['zmf_score'] == 0] = 600
df['xbf_score'][df['xbf_score'] == 0] = 87.6

# 根据身份证号增加性别和年龄 年龄的计算需根据订单创建日期计算
df['age'] = df['card_id'].map(lambda x: 2018 - int(x[6:10]))
df['sex'] = df['card_id'].map(lambda x: int(x[-2]) % 2)

features_cat = ['check_result', 'result', 'pay_num', 'channel', 'goods_type', 'lease_term', 'type', 'order_type',
                'source', 'phone_book', 'emergency_contact_phone', 'old_level', 'create_hour', 'sex', ]
features_number = ['cost', 'daily_rent', 'price', 'age', 'zmf_score', 'xbf_score', ]

df = df[features_cat + features_number]
for col in df.columns.values:
    if df[col].dtype == 'O':
        df[col].fillna(value='NODATA', inplace=True)
df.fillna(value=0, inplace=True)

plt.hist(df['zmf_score'])
feature_analyse(df, 'result')
feature_analyse(df, 'pay_num')
feature_analyse(df, 'channel')
feature_kdeplot(df, 'zmf_score')


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
