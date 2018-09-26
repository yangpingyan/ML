#!/usr/bin/env python
# coding: utf-8
# @Time : 2018/9/21 17:58
# @Author : yangpingyan@gmail.com


import time
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from lightgbm import LGBMClassifier
from matplotlib.colors import ListedColormap
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict, train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier
from tools_ml import *
# Suppress warnings
import warnings
from scipy.stats import randint

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
if os.getcwd().find('titanic') == -1:
    os.chdir('titanic')

df_train = pd.read_csv("train.csv", encoding='utf-8', engine='python')
df_test = pd.read_csv("test.csv", encoding='utf-8', engine='python')
combine = [df_train, df_test]
print("初始数据量: {}".format(df_train.shape))
target = 'TARGET'
# ## 数据简单计量分析
# 查看头尾数据
df_train
# 所有特征值
df_train.columns.values
# 查看数据类型
df_train.dtypes.value_counts()
# 缺失值比率
missing_values_table(df_train)
# 特征中不同值得个数
df_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)
#  数值描述
df_train.describe()
# 类别描述
df_train.describe(include='O')

# 分析特征
feature_analyse(df_train, 'Pclass', 'Survived')
# 特征选择
# f_classif
# mutual_info_classif
# 丢弃不需要的特征
df_train.drop(['Ticket', 'Cabin'], axis=1, inplace=True)
df_test.drop(['Ticket', 'Cabin'], axis=1, inplace=True)
combine = [df_train, df_test]

for dataset in combine:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(df_train['Title'], df_train['Sex'])

# We can replace many titles with a more common name or classify them as Rare.
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

feature_analyse(df_train, 'Title', 'Survived')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

df_train

df_train = df_train.drop(['Name', 'PassengerId'], axis=1, errors='ignore')
df_test = df_test.drop(['Name'], axis=1, errors='ignore')
combine = [df_train, df_test]
df_train.shape, df_test.shape

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

guess_ages = np.zeros((2, 3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)

# 年龄分类
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

df_train = df_train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
df_test = df_test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [df_train, df_test]

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

embarked_fill_value = df_train['Embarked'].value_counts().index[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(embarked_fill_value)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

df_test['Fare'].fillna(df_train['Fare'].dropna().median(), inplace=True)

df_train['FareBand'] = pd.qcut(df_train['Fare'], 4)
df_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',
                                                                                            ascending=True)
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

df_train = df_train.drop(['FareBand'], axis=1)
combine = [df_train, df_test]

print("保存的数据量: {}".format(df_train.shape))
df_train.to_csv("titanic_ml.csv", index=False)

# 查看各特征关联度
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df_train.astype(float).corr(), linewidths=0.1, vmax=1.0,
            square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)

# 机器学习 模型微调-随机搜索
classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
	AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    SGDClassifier(max_iter=5),
    Perceptron(),
    XGBClassifier()]


label = 'Survived'
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

x = df_train.drop([label], axis=1).values
y = df_train[label].values

acc_dict = {}

for train_index, test_index in sss.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(x_train, y_train)
        train_predictions = clf.predict(x_test)
        acc = accuracy_score(y_test, train_predictions)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc

df_accuracy = pd.DataFrame.from_dict(acc_dict, orient='index', columns=['accuracy'])
df_accuracy.sort_values(by=['accuracy'], ascending=False, inplace=True)
print(df_accuracy)
print("Highest accuracy is {}, model is {}".format(df_accuracy['accuracy'].max(), df_accuracy['accuracy'].idxmax()))

# 预测test.csv
clf = SVC(probability=True)
clf.fit(x, y)
x_submission = df_test.drop("PassengerId", axis=1).copy()
y_submission = clf.predict(x_submission)

submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": y_submission
})
submission.to_csv('submission.csv', index=False)

# kaggle competitions submit -c titanic -f titanic/submission.csv -m "SVC"



# 模型微调-随机搜索
param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=8),
}

forest_clf = RandomForestClassifier()
rnd_search = RandomizedSearchCV(forest_clf, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='roc_auc', n_jobs=-1)
rnd_search.fit(x_train, y_train)
rnd_search.best_params_
rnd_search.best_estimator_
rnd_search.best_score_
cvres = rnd_search.cv_results_

feature_importances = rnd_search.best_estimator_.feature_importances_
importance_df = pd.DataFrame({'name': x_train.columns, 'importance': feature_importances})
importance_df.sort_values(by=['importance'], ascending=False, inplace=True)
print(importance_df)

# 用测试集评估系统
clf = rnd_search.best_estimator_
starttime = time.clock()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
add_score(score_df, clf.__class__.__name__ + 'best', time.clock() - starttime, y_pred, y_test)
print(score_df)
