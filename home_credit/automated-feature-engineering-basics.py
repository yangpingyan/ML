# coding: utf-8

# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# featuretools for automated feature engineering
import featuretools as ft

# matplotlit and seaborn for visualizations
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 22
import seaborn as sns

# Suppress warnings from pandas
import warnings

warnings.filterwarnings('ignore')

# Read in the datasets and limit to the first 1000 rows (sorted by SK_ID_CURR) 
# This allows us to actually see the results in a reasonable amount of time! 
app_train = pd.read_csv(r"d:\datasets_ml\home_credit\application_train.csv", encoding='utf-8',
                        engine='python').sort_values(
    'SK_ID_CURR').reset_index(drop=True).loc[:1000, :]
app_test = pd.read_csv(r'd:\datasets_ml\home_credit\application_test.csv').sort_values('SK_ID_CURR').reset_index(
    drop=True).loc[:1000, :]
bureau = pd.read_csv(r'd:\datasets_ml\home_credit\bureau.csv').sort_values(
    ['SK_ID_CURR', 'SK_ID_BUREAU']).reset_index(drop=True).loc[:1000, :]
bureau_balance = pd.read_csv(r'd:\datasets_ml\home_credit\bureau_balance.csv').sort_values(
    'SK_ID_BUREAU').reset_index(drop=True).loc[:1000, :]
cash = pd.read_csv(r'd:\datasets_ml\home_credit\POS_CASH_balance.csv').sort_values(
    ['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop=True).loc[:1000, :]
credit = pd.read_csv(r'd:\datasets_ml\home_credit\credit_card_balance.csv').sort_values(
    ['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop=True).loc[:1000, :]
previous = pd.read_csv(r'd:\datasets_ml\home_credit\previous_application.csv').sort_values(
    ['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop=True).loc[:1000, :]
installments = pd.read_csv(r'd:\datasets_ml\home_credit\installments_payments.csv').sort_values(
    ['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop=True).loc[:1000, :]

# Add identifying column
app_train['set'] = 'train'
app_test['set'] = 'test'
app_test["TARGET"] = np.nan

# Append the dataframes
app = app_train.append(app_test, ignore_index=True)

# Entity set with id applications
es = ft.EntitySet(id='clients')

# Entities with a unique index
es = es.entity_from_dataframe(entity_id='app', dataframe=app, index='SK_ID_CURR')
es = es.entity_from_dataframe(entity_id='bureau', dataframe=bureau, index='SK_ID_BUREAU')
es = es.entity_from_dataframe(entity_id='previous', dataframe=previous, index='SK_ID_PREV')

# Entities that do not have a unique index
es = es.entity_from_dataframe(entity_id='bureau_balance', dataframe=bureau_balance,
                              make_index=True, index='bureaubalance_index')

es = es.entity_from_dataframe(entity_id='cash', dataframe=cash,
                              make_index=True, index='cash_index')

es = es.entity_from_dataframe(entity_id='installments', dataframe=installments,
                              make_index=True, index='installments_index')

es = es.entity_from_dataframe(entity_id='credit', dataframe=credit,
                              make_index=True, index='credit_index')

print('Parent: app, Parent Variable: SK_ID_CURR\n\n', app.iloc[:, 111:115].head())
print('\nChild: bureau, Child Variable: SK_ID_CURR\n\n', bureau.iloc[10:30, :4].head())

# Relationship between app and bureau
r_app_bureau = ft.Relationship(es['app']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])

# Relationship between bureau and bureau balance
r_bureau_balance = ft.Relationship(es['bureau']['SK_ID_BUREAU'], es['bureau_balance']['SK_ID_BUREAU'])

# Relationship between current app and previous apps
r_app_previous = ft.Relationship(es['app']['SK_ID_CURR'], es['previous']['SK_ID_CURR'])

# Relationships between previous apps and cash, installments, and credit
r_previous_cash = ft.Relationship(es['previous']['SK_ID_PREV'], es['cash']['SK_ID_PREV'])
r_previous_installments = ft.Relationship(es['previous']['SK_ID_PREV'], es['installments']['SK_ID_PREV'])
r_previous_credit = ft.Relationship(es['previous']['SK_ID_PREV'], es['credit']['SK_ID_PREV'])

# Add in the defined relationships
es = es.add_relationships([r_app_bureau, r_bureau_balance, r_app_previous,
                           r_previous_cash, r_previous_installments, r_previous_credit])
# Print out the EntitySet
es

# List the primitives in a dataframe
primitives = ft.list_primitives()
pd.options.display.max_colwidth = 100
primitives[primitives['type'] == 'aggregation'].head(10)
primitives[primitives['type'] == 'transform'].head(10)

# Default primitives from featuretools
default_agg_primitives = ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]
default_trans_primitives = ["day", "year", "month", "weekday", "haversine", "numwords", "characters"]

# DFS with specified primitives
feature_names = ft.dfs(entityset=es, target_entity='app',
                       trans_primitives=default_trans_primitives,
                       agg_primitives=default_agg_primitives,
                       max_depth=2, features_only=True)

print('%d Total Features' % len(feature_names))

# DFS with default primitives
feature_matrix, feature_names = ft.dfs(entityset=es, target_entity='app',
                                       trans_primitives=default_trans_primitives,
                                       agg_primitives=default_agg_primitives,
                                       max_depth=2, features_only=False, verbose=True)

pd.options.display.max_columns = 1700
feature_matrix.head(10)
feature_names[-20:]

# Specify the aggregation primitives
feature_matrix_spec, feature_names_spec = ft.dfs(entityset=es, target_entity='app',
                                                 agg_primitives=['sum', 'count', 'min', 'max', 'mean', 'mode'],
                                                 max_depth=2, features_only=False, verbose=True)

pd.options.display.max_columns = 1000
feature_matrix_spec.head(10)

correlations = pd.read_csv('../input/home-credit-default-risk-feature-tools/correlations_spec.csv', index_col=0)
correlations.index.name = 'Variable'
correlations.head()

# ### Correlations with the Target

# In[18]:


correlations_target = correlations.sort_values('TARGET')['TARGET']
# Most negative correlations
correlations_target.head()

# In[19]:


# Most positive correlations
correlations_target.dropna().tail()

# Several of the features created by featuretools are among the most correlated with the `TARGET` (in terms of absolute magnitude). However, that does not mean they are necessarily "important".
# 
# 

# ### Visualize Distribution of Correlated Variables
# 
# One way we can look at the resulting features and their relation to the target is with a kernel density estimate plot. This shows the distribution of a single variable, and can be thought of as a smoothed histogram. To show the effect of a categorical variable on the distribution of a numeric variable, we can color the plot by th value of the categorical variable. In the plot below, we show the distribution of two of the newly created features, colored by the value of the target. 
# 
# First, we read in some of the feature matrix using the `nrows` argument of pandas `read_csv` function. This ensures we will not read in the entire 2 GB file. 

# In[20]:


features_sample = pd.read_csv('../input/home-credit-default-risk-feature-tools/feature_matrix.csv', nrows=20000)
features_sample = features_sample[features_sample['set'] == 'train']
features_sample.head()


# In[21]:


def kde_target_plot(df, feature):
    """Kernel density estimate plot of a feature colored
    by value of the target."""

    # Need to reset index for loc to workBU
    df = df.reset_index()
    plt.figure(figsize=(10, 6))
    plt.style.use('fivethirtyeight')

    # plot repaid loans
    sns.kdeplot(df.loc[df['TARGET'] == 0, feature], label='target == 0')
    # plot loans that were not repaid
    sns.kdeplot(df.loc[df['TARGET'] == 1, feature], label='target == 1')

    # Label the plots
    plt.title('Distribution of Feature by Target Value')
    plt.xlabel('%s' % feature);
    plt.ylabel('Density');
    plt.show()


# In[22]:


kde_target_plot(features_sample, feature='MAX(previous_app.MEAN(credit.CNT_DRAWINGS_ATM_CURRENT))')

# The correlation between this feature and the target is extremely weak and could be only noise. Trying to interpret this feature is difficult, but my best guess is: a client's maximum value of average number of atm drawings per month on previous credit card loans. (We are only using a sample of the features, so this might not be representative of the entire dataset).

# Another area to investigate is highly correlated features, known as collinear features. We can look for pairs of correlated features and potentially remove any above a threshold.

# #### Collinear Features

# In[23]:


threshold = 0.9

correlated_pairs = {}

# Iterate through the columns
for col in correlations:
    # Find correlations above the threshold
    above_threshold_vars = [x for x in list(correlations.index[correlations[col] > threshold]) if x != col]
    correlated_pairs[col] = above_threshold_vars

# In[24]:


correlated_pairs['MEAN(credit.AMT_PAYMENT_TOTAL_CURRENT)']

# In[25]:


correlations['MEAN(credit.AMT_PAYMENT_TOTAL_CURRENT)'].sort_values(ascending=False).head()

# In[26]:


plt.plot(features_sample['MEAN(credit.AMT_PAYMENT_TOTAL_CURRENT)'],
         features_sample['MEAN(previous_app.MEAN(credit.AMT_PAYMENT_CURRENT))'], 'bo')
plt.title('Highly Correlated Features');

# These variables all have a 0.99 correlation with each other which is nearly perfectly positively linear. Including them all in the model is unnecessary because it would be encoding redundant information. We would probably want to remove some of these highly correlated variables in order to help the model learn and generalize better.

# ## Feature Importances
# 
# The feature importances returned by a tree-based model [represent the reduction in impurity](https://stackoverflow.com/questions/15810339/how-are-feature-importances-in-randomforestclassifier-determined) from including the feature in the model. While the absolute value of the importances can be difficult to interpret, looking at the relative value of the importances allows us to compare the relevance of features. Although we want to be careful about placing too much value on the feature importances, they can be a useful method for dimensionality reduction and understanding the model.

# In[27]:


# Read in the feature importances and sort with the most important at the top
fi = pd.read_csv('../input/home-credit-default-risk-feature-tools/spec_feature_importances_ohe.csv', index_col=0)
fi = fi.sort_values('importance', ascending=False)
fi.head(15)

# The most important feature created by featuretools was the maximum number of days before current application that the client applied for a loan at another institution. (This feature is originally recorded as negative, so the maximum value would be closest to zero).

# In[28]:


kde_target_plot(features_sample, feature='MAX(bureau.DAYS_CREDIT)')

# We can calculate the number of top 100 features that were made by featuretools.

# In[29]:


# List of the original features (after one-hot)
original_features = list(pd.get_dummies(app).columns)

created_features = []

# Iterate through the top 100 features
for feature in fi['feature'][:100]:
    if feature not in original_features:
        created_features.append(feature)

print('%d of the top 100 features were made by featuretools' % len(created_features))

# Let's write a short function to visualize the 15 most important features.

# In[30]:


import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 22


def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better. 
    
    Parameters
    --------
        df : dataframe
            feature importances. Must have the features in a column
            called `features` and the importances in a column called `importance
        
    Return
    -------
        shows a plot of the 15 most importance features
        
        df : dataframe
            feature importances sorted by importance (highest to lowest) 
            with a column for normalized importance
        """

    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()

    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(14, 10))
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


# In[31]:


fi = plot_feature_importances(fi)

# The most important feature created by featuretools was `MAX(bureau.DAYS_CREDIT)`. `DAYS_CREDIT` represents the number of days before the current application at Home Credit that the applicant applied for a loan at another credit institution. The maximum of this value (over the previous loans) is therefore represented by this feature. We also see several important features with a depth of two such as `MEAN(previous_app.MIN(installments.AMT_PAYMENT))` which is the average over a client's loans of the minimum value of previous credit application installment payments.
# 
# Feature importances can be used for dimensionality reduction. They can also be used to help us better understand a problem. For example, we could use the most important features in order to concentrate on these aspects of a client when evaluating a potential loan. Let's look at the number of features with 0 importance which almost certainly can be removed from the featureset. 

# In[32]:


print('There are %d features with 0 importance' % sum(fi['importance'] == 0.0))

# ## Remove Low Importance Features
# 
# Feature selection is an entire topic by itself, but one thing we can do is remove any features that have only a single unique value or are all null. Featuretools has a default method for doing this available in the `selection` module.

# In[33]:


from featuretools import selection

# Remove features with only one unique value
feature_matrix2 = selection.remove_low_information_features(feature_matrix)

print('Removed %d features' % (feature_matrix.shape[1] - feature_matrix2.shape[1]))

# ## Align Train and Test Sets
# 
# We also want to make sure the train and test sets have the same exact features. We can first one-hot encode the data (we'll have to do this anyway for our model) and then align the dataframes on the columns.

# In[34]:


# Separate out the train and test sets
train = feature_matrix2[feature_matrix2['set'] == 'train']
test = feature_matrix2[feature_matrix2['set'] == 'test']

# One hot encoding
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# Align dataframes on the columns
train, test = train.align(test, join='inner', axis=1)
test = test.drop(columns=['TARGET'])

print('Final Training Shape: ', train.shape)
print('Final Testing Shape: ', test.shape)

# Removing the low information features and aligning the dataframes has left us with 1689 features! Feature selection will certainly play an important role when using featuretools.

#  # Conclusions
# 
# In this notebook we went through a basic implementation of using automated feature engineering with featuretools for the Home Credit Default Risk dataset. Although we did not use the advanced functionality of featuretools, we still were able to create useful features that improved the model's performance in cross validation and on the test set. Moreover, automated feature engineering took a fraction of the time spent manual feature engineering while delivering comparable results. 
# 
# __Even the default set of features in featuretools was able to achieve similar performance to hand-engineered features in less than 10% of the time.__
# __Featuretools demonstrably adds value when included in a data scientist's toolbox.__
# 
# The next steps are to take advantage of the advanced functionality in featuretools combined with domain knowledge to create a more useful set of features. We will look explore [tuning featuretools in an upcoming notebook](https://www.kaggle.com/willkoehrsen/intro-to-tuning-automated-feature-engineering)!

# ## Appendix: GBM Model (Used Across Feature Sets)
# ```python
# def model(features, test_features, encoding = 'ohe', n_folds = 5):
#     
#     """Train and test a light gradient boosting model using
#     cross validation. 
#     
#     Parameters
#     --------
#         features (pd.DataFrame): 
#             dataframe of training features to use 
#             for training a model. Must include the TARGET column.
#         test_features (pd.DataFrame): 
#             dataframe of testing features to use
#             for making predictions with the model. 
#         encoding (str, default = 'ohe'): 
#             method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
#             n_folds (int, default = 5): number of folds to use for cross validation
#         
#     Return
#     --------
#         submission (pd.DataFrame): 
#             dataframe with `SK_ID_CURR` and `TARGET` probabilities
#             predicted by the model.
#         feature_importances (pd.DataFrame): 
#             dataframe with the feature importances from the model.
#         valid_metrics (pd.DataFrame): 
#             dataframe with training and validation metrics (ROC AUC) for each fold and overall.
#         
#     """
#     
#     # Extract the ids
#     train_ids = features['SK_ID_CURR']
#     test_ids = test_features['SK_ID_CURR']
#     
#     # Extract the labels for training
#     labels = features['TARGET']
#     
#     # Remove the ids and target
#     features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
#     test_features = test_features.drop(columns = ['SK_ID_CURR'])
#     
#     
#     # One Hot Encoding
#     if encoding == 'ohe':
#         features = pd.get_dummies(features)
#         test_features = pd.get_dummies(test_features)
#         
#         # Align the dataframes by the columns
#         features, test_features = features.align(test_features, join = 'inner', axis = 1)
#         
#         # No categorical indices to record
#         cat_indices = 'auto'
#     
#     # Integer label encoding
#     elif encoding == 'le':
#         
#         # Create a label encoder
#         label_encoder = LabelEncoder()
#         
#         # List for storing categorical indices
#         cat_indices = []
#         
#         # Iterate through each column
#         for i, col in enumerate(features):
#             if features[col].dtype == 'object':
#                 # Map the categorical features to integers
#                 features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
#                 test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))
# 
#                 # Record the categorical indices
#                 cat_indices.append(i)
#     
#     # Catch error if label encoding scheme is not valid
#     else:
#         raise ValueError("Encoding must be either 'ohe' or 'le'")
#         
#     print('Training Data Shape: ', features.shape)
#     print('Testing Data Shape: ', test_features.shape)
#     
#     # Extract feature names
#     feature_names = list(features.columns)
#     
#     # Convert to np arrays
#     features = np.array(features)
#     test_features = np.array(test_features)
#     
#     # Create the kfold object
#     k_fold = KFold(n_splits = n_folds, shuffle = False, random_state = 50)
#     
#     # Empty array for feature importances
#     feature_importance_values = np.zeros(len(feature_names))
#     
#     # Empty array for test predictions
#     test_predictions = np.zeros(test_features.shape[0])
#     
#     # Empty array for out of fold validation predictions
#     out_of_fold = np.zeros(features.shape[0])
#     
#     # Lists for recording validation and training scores
#     valid_scores = []
#     train_scores = []
#     
#     # Iterate through each fold
#     for train_indices, valid_indices in k_fold.split(features):
#         
#         # Training data for the fold
#         train_features, train_labels = features[train_indices], labels[train_indices]
#         # Validation data for the fold
#         valid_features, valid_labels = features[valid_indices], labels[valid_indices]
#         
#         # Create the model
#         model = lgb.LGBMClassifier(n_estimators=10000, boosting_type = 'goss',
# 				   objective = 'binary', 
#                                    class_weight = 'balanced', learning_rate = 0.05, 
#                                    reg_alpha = 0.1, reg_lambda = 0.1, n_jobs = -1, random_state = 50)
#         
#         # Train the model
#         model.fit(train_features, train_labels, eval_metric = 'auc',
#                   eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
#                   eval_names = ['valid', 'train'], categorical_feature = cat_indices,
#                   early_stopping_rounds = 100, verbose = 200)
#         
#         # Record the best iteration
#         best_iteration = model.best_iteration_
#         
#         # Record the feature importances
#         feature_importance_values += model.feature_importances_ / k_fold.n_splits
#         
#         # Make predictions
#         test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
#         
#         # Record the out of fold predictions
#         out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
#         
#         # Record the best score
#         valid_score = model.best_score_['valid']['auc']
#         train_score = model.best_score_['train']['auc']
#         
#         valid_scores.append(valid_score)
#         train_scores.append(train_score)
#         
#         # Clean up memory
#         gc.enable()
#         del model, train_features, valid_features
#         gc.collect()
#         
#     # Make the submission dataframe
#     submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
#     
#     # Make the feature importance dataframe
#     feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
#     
#     # Overall validation score
#     valid_auc = roc_auc_score(labels, out_of_fold)
#     
#     # Add the overall scores to the metrics
#     valid_scores.append(valid_auc)
#     train_scores.append(np.mean(train_scores))
#     
#     # Needed for creating dataframe of validation scores
#     fold_names = list(range(n_folds))
#     fold_names.append('overall')
#     
#     # Dataframe of validation scores
#     metrics = pd.DataFrame({'fold': fold_names,
#                             'train': train_scores,
#                             'valid': valid_scores}) 
#     
#     return submission, feature_importances, metrics
# ```
