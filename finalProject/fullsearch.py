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
import sys
sys.path.append("..")
sys.path.append("../..")
# In[2]:


#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[3]:

#from finalProject.featureEngineer import fe
from finalProject.bestFeatureBefore import fe
showimage = False

# In[4]:


dfy = pd.read_csv("Ytrain.csv")
dfy = dfy.drop(['Id'], 1)
dfy.columns = ['Y']


dfx = pd.read_csv("Xtrain.csv")



df = pd.concat([dfx, dfy], axis=1)



df=fe(df)
X = df.loc[:, df.columns != 'Y'].values
Y = df['Y'].values
#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
x_train, y_train, = X,Y

# In[51]:


# clf = RandomForestClassifier(n_estimators=800, random_state=0)
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
# acc = round(accuracy_score(y_test, y_pred) * 100, 2)
# acc, round(accuracy_score(clf.predict(x_train), y_train) * 100, 2)

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
    'n_estimators': [700,1000,1500],
    'colsample_bytree': [0.6,0.7,0.8,0.9,1],
    'max_depth': [5,7,10,12,15,30,50],
    'subsample': [0.7, 0.8, 0.9,1],
    "min_child_weight" : [1,2,3,6],
    #"learning_rate": [0.03, 0.05, 0.1,0.16]
    "learning_rate": [0.03,0.05, 0.1,0.16]
}

# In[54]:


lgbparam_grid = {
    'n_estimators': [700,800,1000,1500],
    'colsample_bytree': [0.7, 0.8,0.9],
    'max_depth': [15,20,25,50,75,80,90],
    #"num_leaves": [40,50,100,200],
    "num_leaves": [50,70,100,120,200,300,900,1200],
    "learning_rate" : [0.01,0.05,0.1],
    'min_split_gain': [0.2,0.3, 0.4,0.5],
    'subsample': [0.7, 0.8, 0.9,1],
}
#lgbm=pipeline(model,lgbparam_grid)


cbparams = {'depth': [4,6, 7, 8,9,10],'learning_rate' : [0.03,0.05,0.1, 0.15,0.2,0.3],
         'l2_leaf_reg': [1,4,6,9,11,12,13,14],'iterations': [500,700,900,1100]}


models = [
    XGBClassifier(objective='binary:logistic', silent=True, nthread=2, seed=0, verbosity=0),
    cb.CatBoostClassifier(random_seed=0, silent=True,thread_count=2),
    lgb.LGBMClassifier(random_state=0, silent=True,n_jobs=2),
    XGBClassifier(objective='binary:logistic', silent=True, nthread=2, seed=1, verbosity=0),
    cb.CatBoostClassifier(random_seed=1, silent=True,thread_count=2),
    lgb.LGBMClassifier(random_state=1, silent=True,n_jobs=2),
    XGBClassifier(objective='binary:logistic', silent=True, nthread=2, seed=2, verbosity=0),
    cb.CatBoostClassifier(random_seed=2, silent=True,thread_count=2),
    lgb.LGBMClassifier(random_state=2, silent=True,n_jobs=2),
]
params = [
    xgbparams,
    cbparams,
    lgbparam_grid,
    xgbparams,
    cbparams,
    lgbparam_grid,
    xgbparams,
    cbparams,
    lgbparam_grid,
]
# seeds=[0,1,0,1,0,1]
seeds = [0, 0, 0,1,1,1,1,1,1]
param_combo=[40,200,200,40,200,200,40,200,200]
trainedModels = []


# In[ ]:


def pipeline(model, params, random_state=0,param_comb=40):
    folds = 5

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=param_comb,
                                       scoring='accuracy', n_jobs=8, cv=skf.split(x_train, y_train),
                                       verbose=0, random_state=random_state)

    # Here we go
    start_time = timer(None)  # timing starts from this point for "start_time" variable
    random_search.fit(x_train, y_train)
    timer(start_time)  # timing ends here for "start_time" variable
    return random_search


# In[ ]:


for i in range(len(models)):
    #if(i==0 or i==3):continue
    trainedModels.append(pipeline(models[i],params[i],seeds[i],param_combo[i]))
    random_search=trainedModels[-1]
    print(random_search.best_params_)
    print(random_search.best_score_)
    # y_pred=random_search.predict(x_test)
    # acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    #print(acc)


# estimators=[]
# for i in range(len(trainedModels)):
#     estimators.append((str(i),trainedModels[i].best_estimator_))



# votingC = VotingClassifier(estimators=estimators, voting='soft', n_jobs=4)
# votingC = votingC.fit(x_train, y_train)
# y_pred = votingC.predict(x_test)
# acc = round(accuracy_score(y_test, y_pred) * 100, 2)
# print(acc)


