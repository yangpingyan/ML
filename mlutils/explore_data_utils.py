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
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score


# Function to calculate correlations with the target for a dataframe
def target_corrs(df):
    # List of correlations
    corrs = []

    # Iterate through the columns
    for col in df.columns:
        print(col)
        # Skip the target column
        if col != 'TARGET':
            # Calculate correlation with the target
            corr = df['TARGET'].corr(df[col])

            # Append the list as a tuple
            corrs.append((col, corr))

    # Sort by absolute magnitude of correlations
    corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)

    return corrs


def pairs_plot(plot_data, target='TARGET'):
    # Create the pairgrid object
    grid = sns.PairGrid(data=plot_data, size=3, diag_sharey=False,
                        hue='TARGET',
                        vars=[x for x in list(plot_data.columns) if x != 'TARGET'])
    # Upper is a scatter plot
    grid.map_upper(plt.scatter, alpha=0.2)
    # Diagonal is a histogram
    grid.map_diag(sns.kdeplot)
    # Bottom is density plot
    grid.map_lower(sns.kdeplot, cmap=plt.cm.OrRd_r)
    plt.suptitle('Pairs Plot', size=32, y=1.05)
    plt.legend()
    plt.show()


# 保存所有模型得分
def add_score(score_df, name, y_pred, y_test):
    score_df[name] = [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred),
                      f1_score(y_test, y_pred), confusion_matrix(y_test, y_pred)]

    return score_df


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
def feature_analyse(df, feature, label='TARGET', bins=10):
    print("dtype of {} is {}.".format(feature, df[feature].dtype))
    print(df[feature].notnull().value_counts())
    print("-------------------------------------------")
    print(df[feature].value_counts())
    print("-------------------------------------------")
    if df[feature].dtype != 'object':
        plt.figure()
        df_notnull = df[df[feature].notnull()]
        plt.hist(df_notnull[feature])
        feature_kdeplot(df_notnull, feature, label)

    if df[feature].dtype != 'object':
        feature_band = feature + 'Band'
        df[feature_band] = pd.cut(df[feature], bins).astype(str)
        col_ana = feature_band
    else:
        col_ana = feature

    print(df[col_ana].describe())
    print("-------------------------------------------")
    pass_df = pd.DataFrame({'positive': df[df[label] == 1][col_ana].value_counts()})
    reject_df = pd.DataFrame({'negative': df[df[label] == 0][col_ana].value_counts()})
    all_df = pd.DataFrame({'all': df[col_ana].value_counts()})
    analyse_df = all_df.merge(pass_df, how='outer', left_index=True, right_index=True)
    analyse_df = analyse_df.merge(reject_df, how='outer', left_index=True, right_index=True)
    analyse_df['positive_rate'] = analyse_df['positive'] / analyse_df['all']
    # analyse_df.sort_values(by='positive_rate', inplace=True, ascending=False)
    analyse_df.fillna(value=0, inplace=True)
    print(analyse_df)
    plt.figure()
    plt.bar(analyse_df.index, analyse_df['positive_rate'])
    plt.ylabel('Positive Rate')
    plt.title('Positive Rate of ' + feature)
    if df[feature].dtype != 'object':
        df.drop([feature_band], axis=1, errors='ignore', inplace=True)
    plt.show()


# KDE plot
def feature_kdeplot(df, feature, label='check_result'):
    plt.figure()
    sns.kdeplot(df.loc[df[label] == 0, feature], label='0')
    sns.kdeplot(df.loc[df[label] == 1, feature], label='1')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.title('Distribution of ' + feature)
    plt.show()


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
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importances')
    plt.show()

    return df
