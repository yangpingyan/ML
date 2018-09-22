#!/usr/bin/env python
# coding: utf-8
# @Time : 2018/9/21 17:58
# @Author : yangpingyan@gmail.com

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
from tools_ml import *

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
# PROJECT_ROOT_DIR = os.getcwd()
# dataset_path = "datasets"
# DATASETS_PATH = os.path.join(PROJECT_ROOT_DIR, dataset_path, "train.csv")


df_alldata = pd.read_csv("train.csv", encoding='utf-8', engine='python')
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

label = 'Survived'
feature = 'Embarked'
print(df[feature].value_counts())
plt.figure()
if df[feature].dtype != 'object':
    plt.hist(df[feature])
    feature_kdeplot(df, feature, label)
feature_analyse(df, feature, label)


df['Embarked'].fillna(value=df['Embarked'].value_counts().index[0], inplace=True)
df['Age'].fillna(value=df['Age'].value_counts().index[0], inplace=True)



features = ['Survived', 'Pclass',  'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
# 类别特征全部转换成数字
for feature in features:
    if df[feature].dtype == 'object':
        df[feature] = LabelEncoder().fit_transform(df[feature])

df = df[features]
print("保存的数据量: {}".format(df.shape))
df.to_csv("titanic_ml.csv", index=False)

# 查看各特征关联度
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df.astype(float).corr(), linewidths=0.1, vmax=1.0,
            square=True, cmap=plt.cm.RdBu, linecolor='white', annot=True)
