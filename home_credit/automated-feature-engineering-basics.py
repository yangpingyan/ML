
# coding: utf-8

# # Introduction: Automated Feature Engineering Basics
# 
# In this notebook, we will walk through applying automated feature engineering to the [Home Credit Default Risk dataset](https://www.kaggle.com/c/home-credit-default-risk) using the featuretools library. [Featuretools](https://docs.featuretools.com/) is an open-source Python package for automatically creating new features from multiple tables of structured, related data. It is ideal tool for problems such as the Home Credit Default Risk competition where there are several related tables that need to be combined into a single dataframe for training (and one for testing). 
# 
# ## Feature Engineering
# 
# The objective of [feature engineering](https://en.wikipedia.org/wiki/Feature_engineering) is to create new features (alos called explantory variables or predictors) to represent as much information from an entire dataset in one table.  Typically, this process is done by hand using pandas operations such as `groupby`, `agg`, or `merge` and can be very tedious. Moreover, manual feature engineering is limited both by human time constraints and imagination: we simply cannot conceive of every possible feature that will be useful. (For an example of using manual feature engineering, check out [part one](https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering) and [part two](https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering-p2) applied to this competition). The importance of creating the proper features cannot be overstated because a machine learning model can only learn from the data we give to it. Extracting as much information as possible from the available datasets is crucial to creating an effective solution.
# 
# [Automated feature engineering](https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219) aims to help the data scientist with the problem of feature creation by automatically building hundreds or thousands of new features from a dataset. Featuretools - the only library for automated feature engineering at the moment - will not replace the data scientist, but it will allow her to focus on more valuable parts of the machine learning pipeline, such as delivering robust models into production. 
# 
# Here we will touch on the concepts of automated feature engineering with featuretools and show how to implement it for the Home Credit Default Risk competition. We will stick to the basics so we can get the ideas down and then build upon this foundation in later work when we customize featuretools. We will work with a subset of the data because this is a computationally intensive job that is outside the capabilities of the Kaggle kernels. I took the work done in this notebook and ran the methods on the entire dataset with the results [available here](https://www.kaggle.com/willkoehrsen/home-credit-default-risk-feature-tools). At the end of this notebook, we'll look at the features themselves, as well as the results of modeling with different combinations of hand designed and automatically built features. 
# 
# If you are new to this competition, I suggest checking out [this post to get started](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction). For a good take on why features are so important, here's a [blog post](https://www.featurelabs.com/blog/secret-to-data-science-success/) by one of the developers of Featuretools. 

# In[1]:


# Uncomment and run if kernel does not already have featuretools
# !pip install featuretools


# In[2]:


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


# # Problem
# 
# The Home Credit Default Risk competition is a supervised classification machine learning task. The objective is to use historical financial and socioeconomic data to predict whether or not an applicant will be able to repay a loan. This is a standard supervised classification task:
# 
# * __Supervised__: The labels are included in the training data and the goal is to train a model to learn to predict the labels from the features
# * __Classification__: The label is a binary variable, 0 (will repay loan on time), 1 (will have difficulty repaying loan)
# 
# ## Dataset
# 
# The data is provided by [Home Credit](http://www.homecredit.net/about-us.aspx), a service dedicated to provided lines of credit (loans) to the unbanked population. 
# 
# There are 7 different data files:
# 
# * __application_train/application_test__: the main training and testing data with information about each loan application at Home Credit. Every loan has its own row and is identified by the `SK_ID_CURR`. The training application data comes with the `TARGET` with indicating 0: the loan was repaid and 1: the loan was not repaid. 
# * __bureau__: data concerning client's previous credits from other financial institutions. Each previous credit has its own row in bureau and is identified by the `SK_ID_BUREAU`, Each loan in the application data can have multiple previous credits.
# * __bureau_balance__: monthly data about the previous credits in bureau. Each row is one month of a previous credit, and a single previous credit can have multiple rows, one for each month of the credit length. 
# * __previous_application__: previous applications for loans at Home Credit of clients who have loans in the application data. Each current loan in the application data can have multiple previous loans. Each previous application has one row and is identified by the feature `SK_ID_PREV`. 
# * __POS_CASH_BALANCE__: monthly data about previous point of sale or cash loans clients have had with Home Credit. Each row is one month of a previous point of sale or cash loan, and a single previous loan can have many rows.
# * __credit_card_balance__: monthly data about previous credit cards clients have had with Home Credit. Each row is one month of a credit card balance, and a single credit card can have many rows.
# * __installments_payment__: payment history for previous loans at Home Credit. There is one row for every made payment and one row for every missed payment. 
# 
# The diagram below (provided by Home Credit) shows how the tables are related. This will be very useful when we need to define relationships in featuretools. 
# 
# ![image](https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png)
# 
# ### Read in Data and Create Small Datasets
# 
# We will read in the full dataset, sort by the `SK_ID_CURR` and keep only the first 1000 rows to make the calculations feasible. Later we can convert to a script and run with the entire datasets.

# In[3]:


# Read in the datasets and limit to the first 1000 rows (sorted by SK_ID_CURR) 
# This allows us to actually see the results in a reasonable amount of time! 
app_train = pd.read_csv('../input/home-credit-default-risk/application_train.csv').sort_values('SK_ID_CURR').reset_index(drop = True).loc[:1000, :]
app_test = pd.read_csv('../input/home-credit-default-risk/application_test.csv').sort_values('SK_ID_CURR').reset_index(drop = True).loc[:1000, :]
bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv').sort_values(['SK_ID_CURR', 'SK_ID_BUREAU']).reset_index(drop = True).loc[:1000, :]
bureau_balance = pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv').sort_values('SK_ID_BUREAU').reset_index(drop = True).loc[:1000, :]
cash = pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:1000, :]
credit = pd.read_csv('../input/home-credit-default-risk/credit_card_balance.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:1000, :]
previous = pd.read_csv('../input/home-credit-default-risk/previous_application.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:1000, :]
installments = pd.read_csv('../input/home-credit-default-risk/installments_payments.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:1000, :]


# We'll join the train and test set together but add a separate column identifying the set. This is important because we are going to want to apply the same exact procedures to each dataset. It's safest to just join them together and treat them as a single dataframe. 
# 
# (I'm not sure if this is allowing data leakage into the train set and if these feature creation operations should be applied separately. Any thoughts would be much appreciated!)

# In[4]:


# Add identifying column
app_train['set'] = 'train'
app_test['set'] = 'test'
app_test["TARGET"] = np.nan

# Append the dataframes
app = app_train.append(app_test, ignore_index = True)


# # Featuretools Basics
# 
# [Featuretools](https://docs.featuretools.com/#minute-quick-start) is an open-source Python library for automatically creating features out of a set of related tables using a technique called [deep feature synthesis](http://www.jmaxkanter.com/static/papers/DSAA_DSM_2015.pdf). Automated feature engineering, like many topics in machine learning, is a complex subject built upon a foundation of simpler ideas. By going through these ideas one at a time, we can build up our understanding of how featuretools which will later allow for us to get the most out of it.
# 
# There are a few concepts that we will cover along the way:
# 
# * [Entities and EntitySets](https://docs.featuretools.com/loading_data/using_entitysets.html)
# * [Relationships between tables](https://docs.featuretools.com/loading_data/using_entitysets.html#adding-a-relationship)
# * [Feature primitives](https://docs.featuretools.com/automated_feature_engineering/primitives.html): aggregations and transformations
# * [Deep feature synthesis](https://docs.featuretools.com/automated_feature_engineering/afe.html)
# 
# # Entities and Entitysets
# 
# An entity is simply a table or in Pandas, a `dataframe`. The observations are in the rows and the features in the columns. An entity in featuretools must have a unique index where none of the elements are duplicated.  Currently, only `app`, `bureau`, and `previous` have unique indices (`SK_ID_CURR`, `SK_ID_BUREAU`, and `SK_ID_PREV` respectively). For the other dataframes, we must pass in `make_index = True` and then specify the name of the index. Entities can also have time indices where each entry is identified by a unique time. (There are not datetimes in any of the data, but there are relative times, given in months or days, that we could consider treating as time variables).
# 
# An [EntitySet](https://docs.featuretools.com/loading_data/using_entitysets.html) is a collection of tables and the relationships between them. This can be thought of a data structute with its own methods and attributes. Using an EntitySet allows us to group together multiple tables and manipulate them much quicker than individual tables. 
# 
# First we'll make an empty entityset named clients to keep track of all the data.

# In[5]:


# Entity set with id applications
es = ft.EntitySet(id = 'clients')


# Now we define each entity, or table of data. We need to pass in an index if the data has one or `make_index = True` if not. Featuretools will automatically infer the types of variables, but we can also change them if needed. For intstance, if we have a categorical variable that is represented as an integer we might want to let featuretools know the right type.

# In[6]:


# Entities with a unique index
es = es.entity_from_dataframe(entity_id = 'app', dataframe = app, index = 'SK_ID_CURR')

es = es.entity_from_dataframe(entity_id = 'bureau', dataframe = bureau, index = 'SK_ID_BUREAU')

es = es.entity_from_dataframe(entity_id = 'previous', dataframe = previous, index = 'SK_ID_PREV')

# Entities that do not have a unique index
es = es.entity_from_dataframe(entity_id = 'bureau_balance', dataframe = bureau_balance, 
                              make_index = True, index = 'bureaubalance_index')

es = es.entity_from_dataframe(entity_id = 'cash', dataframe = cash, 
                              make_index = True, index = 'cash_index')

es = es.entity_from_dataframe(entity_id = 'installments', dataframe = installments,
                              make_index = True, index = 'installments_index')

es = es.entity_from_dataframe(entity_id = 'credit', dataframe = credit,
                              make_index = True, index = 'credit_index')


# # Relationships
# 
# Relationships are a fundamental concept not only in featuretools, but in any relational database. The best way to think of a one-to-many relationship is with the analogy of parent-to-child. A parent is a single individual, but can have mutliple children. The children can then have multiple children of their own. In a _parent table_, each individual has a single row. Each individual in the parent table can have multiple rows in the _child table_. 
# 
# As an example, the `app` dataframe has one row for each client  (`SK_ID_CURR`) while the `bureau` dataframe has multiple previous loans (`SK_ID_PREV`) for each parent (`SK_ID_CURR`). Therefore, the `bureau` dataframe is the child of the `app` dataframe. The `bureau` dataframe in turn is the parent of `bureau_balance` because each loan has one row in `bureau` but multiple monthly records in `bureau_balance`. 

# In[7]:


print('Parent: app, Parent Variable: SK_ID_CURR\n\n', app.iloc[:, 111:115].head())
print('\nChild: bureau, Child Variable: SK_ID_CURR\n\n', bureau.iloc[10:30, :4].head())


# The `SK_ID_CURR` "100002" has one row in the parent table and multiple rows in the child. 
# 
# Two tables are linked via a shared variable. The `app` and `bureau` dataframe are linked by the `SK_ID_CURR` variable while the `bureau` and `bureau_balance` dataframes are linked with the `SK_ID_BUREAU`. Defining the relationships is relatively straightforward, and the diagram provided by the competition is helpful for seeing the relationships. For each relationship, we need to specify the parent variable and the child variable. Altogether, there are a total of 6 relationships between the tables. Below we specify all six relationships and then add them to the EntitySet.

# In[8]:


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


# In[9]:


# Add in the defined relationships
es = es.add_relationships([r_app_bureau, r_bureau_balance, r_app_previous,
                           r_previous_cash, r_previous_installments, r_previous_credit])
# Print out the EntitySet
es


# Slightly advanced note: we need to be careful to not create a [diamond graph](https://en.wikipedia.org/wiki/Diamond_graph) where there are multiple paths from a parent to a child. If we directly link `app` and `cash` via `SK_ID_CURR`; `previous` and `cash` via `SK_ID_PREV`; and `app` and `previous` via `SK_ID_CURR`, then we have created two paths from `app` to `cash`. This results in ambiguity, so the approach we have to take instead is to link `app` to `cash` through `previous`. We establish a relationship between `previous` (the parent) and `cash` (the child) using `SK_ID_PREV`. Then we establish a relationship between `app` (the parent) and `previous` (now the child) using `SK_ID_CURR`. Then featuretools will be able to create features on `app` derived from both `previous` and `cash` by stacking multiple primitives. 

# All entities in the entity can be related to each other. In theory this allows us to calculate features for any of the entities, but in practice, we will only calculate features for the `app` dataframe since that will be used for training/testing. 

# # Feature Primitives
# 
# A [feature primitive](https://docs.featuretools.com/automated_feature_engineering/primitives.html) is an operation applied to a table or a set of tables to create a feature. These represent simple calculations, many of which we already use in manual feature engineering, that can be stacked on top of each other to create complex features. Feature primitives fall into two categories:
# 
# * __Aggregation__: function that groups together child datapoints for each parent and then calculates a statistic such as mean, min, max, or standard deviation. An example is calculating the maximum previous loan amount for each client. An aggregation works across multiple tables using relationships between tables.
# * __Transformation__: an operation applied to one or more columns in a single table. An example would be taking the absolute value of a column, or finding the difference between two columns in one table.
# 
# A list of the available features primitives in featuretools can be viewed below.

# In[10]:


# List the primitives in a dataframe
primitives = ft.list_primitives()
pd.options.display.max_colwidth = 100
primitives[primitives['type'] == 'aggregation'].head(10)


# In[11]:


primitives[primitives['type'] == 'transform'].head(10)


# # Deep Feature Synthesis
# 
# Deep Feature Synthesis (DFS) is the process featuretools uses to make new features. DFS stacks feature primitives to form features with a "depth" equal to the number of primitives. For example, if we take the maximum value of a client's previous loans (say `MAX(previous.loan_amount)`), that is a "deep feature" with a depth of 1. To create a feature with a depth of two, we could stack primitives by taking the maximum value of a client's average montly payments per previous loan (such as `MAX(previous(MEAN(installments.payment)))`). The [original paper on automated feature engineering using deep feature synthesis](https://dai.lids.mit.edu/wp-content/uploads/2017/10/DSAA_DSM_2015.pdf) is worth a read. 
# 
# To perform DFS in featuretools, we use the `dfs`  function passing it an `entityset`, the `target_entity` (where we want to make the features), the `agg_primitives` to use, the `trans_primitives` to use and the `max_depth` of the features. Here we will use the default aggregation and transformation primitives,  a max depth of 2, and calculate primitives for the `app` entity. Because this process is computationally expensive, we can run the function using `features_only = True` to return only a list of the features and not calculate the features themselves. This can be useful to look at the resulting features before starting an extended computation.

# ### DFS with Default Primitives

# In[12]:


# Default primitives from featuretools
default_agg_primitives =  ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]
default_trans_primitives =  ["day", "year", "month", "weekday", "haversine", "numwords", "characters"]

# DFS with specified primitives
feature_names = ft.dfs(entityset = es, target_entity = 'app',
                       trans_primitives = default_trans_primitives,
                       agg_primitives=default_agg_primitives, 
                       max_depth = 2, features_only=True)

print('%d Total Features' % len(feature_names))


# If you are interested in running this call on the entire dataset and making the features, I wrote a script [for that here](https://www.kaggle.com/willkoehrsen/feature-engineering-using-feature-tools). Unfortunately, this will not run in a Kaggle kernel due to the computational expense of the operation. Using a computer with 64GB of ram, this function call took around 24 hours (I don't think I'm technically breaking the rules of my university's high powered computing center). I have made the entire dataset available [here](https://www.kaggle.com/willkoehrsen/home-credit-default-risk-feature-tools/data) in the file called `feature_matrix.csv`. 
# 
# To generate a subset of the features, run the code cell below.

# In[13]:


# DFS with default primitives
feature_matrix, feature_names = ft.dfs(entityset = es, target_entity = 'app',
                                       trans_primitives = default_trans_primitives,
                                       agg_primitives=default_agg_primitives, 
                                        max_depth = 2, features_only=False, verbose = True)

pd.options.display.max_columns = 1700
feature_matrix.head(10)


# In[14]:


feature_names[-20:]


# ### DFS with Selected Aggregation Primitives
# 
# With featuretools, we were able to go from 121 original features to almost 1700 in a few lines of code.  When I did feature engineering by hand, it took about 12 hours to create a comparable size dataset. However, while we get a lot of features in featuretools, this function call is not very well-informed. We simply used the default aggregations without thinking about which ones are "important" for the problem. We end up with a lot of features, but they are probably not all relevant to the problem. Too many irrelevant features can decrease performance by drowning out the important features (related to the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality))
# 
# The next call we make will specify a smaller set of features. We still are not using much domain knowledge, but this feature set will be more manageable. The next step from here is improving the features we actually build and performing feature selection.

# In[15]:


# Specify the aggregation primitives
feature_matrix_spec, feature_names_spec = ft.dfs(entityset = es, target_entity = 'app',  
                                                 agg_primitives = ['sum', 'count', 'min', 'max', 'mean', 'mode'], 
                                                 max_depth = 2, features_only = False, verbose = True)


# That "only" gives us 884 features (and takes about 12 hours to run on the complete dataset). 

# In[16]:


pd.options.display.max_columns = 1000
feature_matrix_spec.head(10)


# ## Notes on Basic Implementation
# 
# These calls represent only a [small fraction of the ability of featuretools](https://docs.featuretools.com/guides/tuning_dfs.html). We did not specify the variable types when creating entities, did not use the relative time variables, and didn't touch on [custom primitives](https://docs.featuretools.com/guides/advanced_custom_primitives.html) or seed features or interesting values! Nonetheless, in this notebook, we were able to learn the basic foundations which will allow us to more effective use the tool as we learn how it works.  Now, let's take a look at some of the features we have built and modeling results.

# # Results
# 
# To determine whether our basic implementation of featuretools was useful, we can look at several results:
# 
# * Cross validation scores and public leaderboard scores using several different sets of features.
# * Correlations: both between the features and the `TARGET`, and between features themselves
# * Feature importances: determined by a gradient boosting machine model
# 

# ## Feature Performance Experiments
# 
# To compare a number of different feature sets for the machine learning task, I set up several experiments.. In order to isolate the effect of the features, the same model was used to test a number of different feature sets. The model (which can be viewed in the appendix) is a basic LightGBM algorithm using 5-fold cross validation for training and evaluation. First, we establish a control dataset, and then we carry out a series of experiments and present the results.
# 
# * Control: using only data from the `application` dataset
# * Test One: manual feature engineering using only the `application`, `bureau` and `bureau_balance` data
# * Test Two: manual feature engineering using all datasets
# * Test Three: featuretools default features (in the `feature_matrix`)
# * Test Four: featuretools specified features (in the `feature_matrix_spec`)
# * Test Five: featuretools specified features combined with manual feature engineering 
# 
# The number of features is after one-hot encoding, the validation receiver operating characteristic area under the curve (ROC AUC) is calculated using 5-fold cross validation, the test ROC AUC is from the public leaderboard, and the time spent designing is my best estimate of how long it took to make the dataset! 
# 
# | Test    | Number of Features | Validation ROC AUC | Test ROC AUC | Time Spent |
# |---------|--------------------|--------------------|--------------|--------|
# | Control | 241                |           0.760         |     0.745         |       0.25 hours  |
# | One     | 421                |       0.766             |      0.757        |        8 hours        |
# | Two     |      1465             |          0.785          |         0.783     |                 12 hours |
# | Three   | 1803               |      0.784              |       0.777       |               1 hour
# | Four    | 1156               |         0.786           |        0.779      |                 1.25 hours |
# | Five    |  1624                  |           0. 787        |      0.782        |                    13.25 hours |
# 
# 
# It's hard to say which set is exactly the best (although I trust the cross validation scores more than the public leaderboard) but there are huge discrepancies is the time for development. The specified featuretools dataset was able to achieve nearly the same performance as the hand engineered features on the test set with 8% of the time invested. It's clear that featuretools delivered value on this problem, but it still did not leave us without a job. The vital role of the data scientist now comes down to choosing the correct set of primitives and selecting the best features from among all the candidates. 

# ## Correlations
# 
# Next we can look at correlations within the data. When we look at correlations with the target, we need to be careful about the [multiple comparisons problem](https://towardsdatascience.com/the-multiple-comparisons-problem-e5573e8b9578): if we make a ton of features, some are likely to be correlated with the target simply because of random noise. Using correlations is fine as a first approximation for identifying "good features", but it is not a rigorous feature selection method.  
# 
# Also, based on examining some of the features, it seems there might be issues with [collinearity between features](https://en.wikipedia.org/wiki/Multicollinearity) made by featuretools. Features that are highly correlated with one another can diminish interpretability and generalization performance on the test set. In an ideal scenario, we would have a set of independent features, but that rarely occurs in practice. If there are very highly correlated varibables, we might want to think about removing some of them.
# 
# For the correlations, we will focus on the `feature_matrix_spec`, the features we made by specifying the primitives. The same analysis could be applied to the default feature set. These correlations were calculated using the entire training section of the feature matrix.

# In[17]:


correlations = pd.read_csv('../input/home-credit-default-risk-feature-tools/correlations_spec.csv', index_col = 0)
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


features_sample = pd.read_csv('../input/home-credit-default-risk-feature-tools/feature_matrix.csv', nrows = 20000)
features_sample = features_sample[features_sample['set'] == 'train']
features_sample.head()


# In[21]:


def kde_target_plot(df, feature):
    """Kernel density estimate plot of a feature colored
    by value of the target."""
    
    # Need to reset index for loc to workBU
    df = df.reset_index()
    plt.figure(figsize = (10, 6))
    plt.style.use('fivethirtyeight')
    
    # plot repaid loans
    sns.kdeplot(df.loc[df['TARGET'] == 0, feature], label = 'target == 0')
    # plot loans that were not repaid
    sns.kdeplot(df.loc[df['TARGET'] == 1, feature], label = 'target == 1')
    
    # Label the plots
    plt.title('Distribution of Feature by Target Value')
    plt.xlabel('%s' % feature); plt.ylabel('Density');
    plt.show()


# In[22]:


kde_target_plot(features_sample, feature = 'MAX(previous_app.MEAN(credit.CNT_DRAWINGS_ATM_CURRENT))')


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


plt.plot(features_sample['MEAN(credit.AMT_PAYMENT_TOTAL_CURRENT)'], features_sample['MEAN(previous_app.MEAN(credit.AMT_PAYMENT_CURRENT))'], 'bo')
plt.title('Highly Correlated Features');


# These variables all have a 0.99 correlation with each other which is nearly perfectly positively linear. Including them all in the model is unnecessary because it would be encoding redundant information. We would probably want to remove some of these highly correlated variables in order to help the model learn and generalize better. 

# ## Feature Importances
# 
# The feature importances returned by a tree-based model [represent the reduction in impurity](https://stackoverflow.com/questions/15810339/how-are-feature-importances-in-randomforestclassifier-determined) from including the feature in the model. While the absolute value of the importances can be difficult to interpret, looking at the relative value of the importances allows us to compare the relevance of features. Although we want to be careful about placing too much value on the feature importances, they can be a useful method for dimensionality reduction and understanding the model.

# In[27]:


# Read in the feature importances and sort with the most important at the top
fi = pd.read_csv('../input/home-credit-default-risk-feature-tools/spec_feature_importances_ohe.csv', index_col = 0)
fi = fi.sort_values('importance', ascending = False)
fi.head(15)


# The most important feature created by featuretools was the maximum number of days before current application that the client applied for a loan at another institution. (This feature is originally recorded as negative, so the maximum value would be closest to zero).

# In[28]:


kde_target_plot(features_sample, feature = 'MAX(bureau.DAYS_CREDIT)')


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
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (14, 10))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
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

print('Removed %d features' % (feature_matrix.shape[1]- feature_matrix2.shape[1]))


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
train, test = train.align(test, join = 'inner', axis = 1)
test = test.drop(columns = ['TARGET'])

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
