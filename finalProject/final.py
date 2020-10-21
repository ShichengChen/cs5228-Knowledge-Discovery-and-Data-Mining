# coding: utf-8

# In[1]:


# data manipulation
import pandas as pd
import numpy as np
import math
# visualization
import seaborn as sb
import matplotlib.pyplot as plt

#get_ipython().run_line_magic('matplotlib', 'inline')

# model training
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# model evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# classifiers
from sklearn.naive_bayes import GaussianNB  # naive bayes
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.linear_model import LogisticRegression  # logistic regression
from sklearn.tree import DecisionTreeClassifier  # decision Tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import lightgbm as lgb
from datetime import datetime
# ignore warnings
import warnings
from sklearn.ensemble import VotingClassifier

warnings.filterwarnings('ignore')

# In[2]:


#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import sys
sys.path.append("..")
sys.path.append("../..")
# In[3]:


showimage = False

# In[4]:


dfy = pd.read_csv("Ytrain.csv")
dfy = dfy.drop(['Id'], 1)
dfy.columns = ['Y']

# In[5]:


dfx = pd.read_csv("Xtrain.csv")

# In[6]:


df = pd.concat([dfx, dfy], axis=1)

from finalProject.featureEngineer import fe

df=fe(df)
X = df.loc[:, df.columns != 'Y'].values
Y = df['Y'].values
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# In[51]:


clf = RandomForestClassifier(n_estimators=800, random_state=0)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc = round(accuracy_score(y_test, y_pred) * 100, 2)
acc, round(accuracy_score(clf.predict(x_train), y_train) * 100, 2)

# In[ ]:


import catboost as cb


# In[52]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# In[53]:


xgbparams = {
    'n_estimators': [400, 700, 1000],
    'colsample_bytree': [0.7, 0.8,0.9],
    'max_depth': [10,20,30,40,50],
#     'reg_alpha': [1.1, 1.2, 1.3],
#     'reg_lambda': [1.1, 1.2, 1.3],
    'subsample': [0.7, 0.8, 0.9],
    "min_child_weight" : [1,3,6],
    "learning_rate": [0.05, 0.1,0.16]
}

# In[54]:


lgbparam_grid = {
    'n_estimators': [400, 700, 1000],
    'colsample_bytree': [0.7, 0.8,0.9],
    'max_depth': [15,20,25],
    "num_leaves": [50,100,200,300,900,1200],
#     'reg_alpha': [1.1, 1.2, 1.3],
#     'reg_lambda': [1.1, 1.2, 1.3],
    'min_split_gain': [0.2,0.3, 0.4],
    'subsample': [0.7, 0.8, 0.9],
    'subsample_freq': [20]
}
# lgbm=pipeline(model,lgbparam_grid)


# In[55]:


cbparams = {'depth': [4, 7, 10],
          'learning_rate' : [0.03, 0.1, 0.15],
         'l2_leaf_reg': [1,4,9],
         'iterations': [300,400,500]}

# In[56]:


models = [
    XGBClassifier(objective='binary:logistic', silent=True, nthread=2, seed=0, verbosity=0),
    cb.CatBoostClassifier(random_seed=0, silent=True),
    lgb.LGBMClassifier(random_state=0, silent=True),
    XGBClassifier(objective='binary:logistic', silent=True, nthread=2, seed=1, verbosity=0),
    cb.CatBoostClassifier(random_seed=1, silent=True),
    lgb.LGBMClassifier(random_state=1, silent=True),
]
params = [
    xgbparams,
    cbparams,
    lgbparam_grid,
    xgbparams,
    cbparams,
    lgbparam_grid,
]
# seeds=[0,1,0,1,0,1]
seeds = [0, 0, 0,1,1,1]
trainedModels = []


# In[ ]:


def pipeline(model, params, random_state=0):
    folds = 3
    param_comb = 30

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=param_comb,
                                       scoring='roc_auc', n_jobs=8, cv=skf.split(x_train, y_train),
                                       verbose=0, random_state=random_state)

    # Here we go
    start_time = timer(None)  # timing starts from this point for "start_time" variable
    random_search.fit(x_train, y_train)
    timer(start_time)  # timing ends here for "start_time" variable
    return random_search


# In[ ]:


for i in range(len(models)):
    trainedModels.append(pipeline(models[i],params[i],seeds[i]))
    random_search=trainedModels[-1]
    print(random_search.best_params_)
    y_pred=random_search.predict(x_test)
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    print(acc)


# In[ ]:


estimators=[]
for i in range(len(trainedModels)):
    estimators.append((str(i),trainedModels[i].best_estimator_))


# In[ ]:


# estimators=[
#     ('xgb',XGBClassifier(objective='binary:logistic', silent=True, nthread=2, seed=0, verbosity=0,
#                  **{'subsample': 0.9, 'n_estimators': 700, 'min_child_weight': 3, 'max_depth': 10, 'learning_rate': 0.05, 'colsample_bytree': 0.7})),
#     ('cb',cb.CatBoostClassifier(random_seed=0,silent=True,
#                          **{'learning_rate': 0.15, 'l2_leaf_reg': 9, 'iterations': 300, 'depth': 7})),
#     ('lgb',lgb.LGBMClassifier(random_state=0,silent = True,
#                       **{'subsample_freq': 20, 'subsample': 0.9, 'reg_lambda': 1.2, 'reg_alpha': 1.1, 'num_leaves': 1200, 'n_estimators': 700, 'min_split_gain': 0.4, 'max_depth': 25, 'colsample_bytree': 0.7})),
# ]

# In[ ]:


votingC = VotingClassifier(estimators=estimators, voting='soft', n_jobs=4)
votingC = votingC.fit(x_train, y_train)
y_pred = votingC.predict(x_test)
acc = round(accuracy_score(y_test, y_pred) * 100, 2)
print(acc)

# In[ ]:


#  Time taken: 0 hours 9 minutes and 53.61 seconds.
# {'subsample': 0.9, 'n_estimators': 700, 'min_child_weight': 3, 'max_depth': 10, 'learning_rate': 0.05, 'colsample_bytree': 0.7}
# 93.05

#  Time taken: 0 hours 4 minutes and 34.76 seconds.
# {'learning_rate': 0.15, 'l2_leaf_reg': 9, 'iterations': 300, 'depth': 7}
# 92.82
#  Time taken: 0 hours 7 minutes and 51.83 seconds.
# {'subsample_freq': 20, 'subsample': 0.9, 'reg_lambda': 1.2, 'reg_alpha': 1.1, 'num_leaves': 1200, 'n_estimators': 700, 'min_split_gain': 0.4, 'max_depth': 25, 'colsample_bytree': 0.7}
# 92.74
# [LightGBM] [Warning] Unknown parameter: verbose_eval
# 93.19


# In[ ]:


#  Time taken: 0 hours 36 minutes and 32.98 seconds.
# {'subsample': 0.9, 'n_estimators': 700, 'min_child_weight': 3, 'max_depth': 10, 'learning_rate': 0.05, 'colsample_bytree': 0.7}
# 93.05

#  Time taken: 0 hours 49 minutes and 33.57 seconds.
# {'learning_rate': 0.15, 'l2_leaf_reg': 9, 'iterations': 300, 'depth': 7}
# 92.82

#  Time taken: 0 hours 35 minutes and 11.3 seconds.
# {'subsample_freq': 20, 'subsample': 0.8, 'num_leaves': 50, 'n_estimators': 700, 'min_split_gain': 0.4, 'max_depth': 15, 'colsample_bytree': 0.8}
# 92.89
# 93.2
