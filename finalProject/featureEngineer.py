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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# In[3]:


showimage = False


def fe(df):
    df["State"][(df["State"].isna())] = 'nan'
    unique, pos = np.unique(df['State'], return_inverse=True)
    counts = np.bincount(pos)
    maxpos = counts.argmax()
    df['State'][df['State'] == 'nan'] = unique[maxpos]

    df["LowDoc"][(df["LowDoc"].notna()) & (df["LowDoc"] != 'Y')] = 'N'
    df["LowDoc"][df["LowDoc"].isna()] = 'nan'
    df["RevLineCr"][(df["RevLineCr"].notna()) & (df["RevLineCr"] != 'Y')] = 'N'
    df["RevLineCr"][df["RevLineCr"].isna()] = 'nan'


    df["DisbursementDate"][df["DisbursementDate"].isna()] = '19-Oct-20'

    # In[11]:


    df['DisbursementDate'] = df["DisbursementDate"].apply(str)
    df['DisbursementDate'] = (df['DisbursementDate'].str.split("-").str[-1])
    print(np.sum(df["DisbursementDate"].isna()))
    df['DisbursementDate']

    # In[12]:


    df['DisbursementDate'] = df['DisbursementDate'].astype(float)
    df['DisbursementDate'] = df['DisbursementDate'].astype(int)

    # In[13]:


    df['DisbursementDate'] = df['DisbursementDate'].apply(lambda x: x + 2000 - 2000 if x < 25 else x + 1900 - 2000)

    # In[14]:


    if showimage:
        sb.countplot(x='DisbursementDate', data=df)
        plt.show()

    # In[15]:


    df['ApprovalFY'] = df['ApprovalFY'].str.replace(r'\D', '')
    df["ApprovalFY"][df["ApprovalFY"].isna()] = df["ApprovalFY"][df["ApprovalFY"].notna()].median()

    df['ApprovalFY'] = df['ApprovalFY'].astype(int) - 2000

    # In[16]:


    dollor = ['DisbursementGross', 'GrAppv', 'SBA_Appv']
    for i in dollor:
        df[i] = df[i].apply(str)
        df[i] = df[i].str.strip("$")
        df[i] = df[i].str.replace(",", "")
        df[i] = df[i].str.replace(".", "")
        df[i] = df[i].astype(int) // 100

    # In[17]:


    df['dif'] = df['GrAppv'] - df['SBA_Appv']

    # In[18]:


    # df['NAICS_Sector'] = df[df['NAICS']!=0].NAICS.apply(lambda d: d // 10 ** (int(math.log(d, 10)) - 1))
    # df.loc[df['NAICS'] == 0, 'NAICS_Sector'] = 0
    # df['NAICS_Sector'] = df['NAICS_Sector'].astype('int64')


    # In[19]:


    if showimage:
        sb.clustermap(df.corr(), annot=True)
        plt.show()

    # In[20]:


    df.info()

    # In[21]:


    if showimage:
        sb.distplot(df["NAICS"])
        plt.show()

    # In[22]:


    # df["NoEmp"]=np.log(1+df["NoEmp"])
    if showimage:
        sb.distplot(df["Term"])
        plt.show()

    # In[23]:


    df["NoEmp"] = np.log(1 + df["NoEmp"])
    if showimage:
        sb.distplot(df["NoEmp"])
        plt.show()

    # In[24]:

    print(np.sum(df['NewExist'].isna()))
    df['NewExist'][df['NewExist'].isna()] = 0

    # In[25]:


    if showimage:
        sb.distplot(df["NewExist"])
        plt.show()

    # In[26]:


    df["CreateJob"] = np.log(1 + df["CreateJob"])
    if showimage:
        sb.distplot(df["CreateJob"])
        plt.show()

    # In[27]:


    df["RetainedJob"] = np.log(1 + df["RetainedJob"])
    if showimage:
        sb.distplot(np.log(1 + df["RetainedJob"]))
        plt.show()

    # In[28]:


    np.sum((df["UrbanRural"] == 0) | (df["UrbanRural"] == 1) | (df["UrbanRural"] == 2))

    # In[29]:


    if showimage:
        sb.countplot(x='UrbanRural', data=df)
        plt.show()

    # In[30]:




    # In[31]:


    print(np.sum(~df['FranchiseCode'].isin(['0', '1'])))
    #df['FranchiseCode'][df['FranchiseCode'] == 1] = 0
    df['FranchiseCode'][~df['FranchiseCode'].isin(['0', '1'])] = 2

    # In[32]:


    if showimage:
        sb.distplot(df["FranchiseCode"])
        plt.show()

    # In[33]:

    if showimage and 'Y' in df.columns.values:
        sb.countplot(x='Y', data=df)
        plt.show()

        # In[34]:


    df['EMI'] = df['DisbursementGross'].astype(float) / (df['Term'] + 1)
    df['SequBS'] = (df['State'] == df['BankState'])
    df['portion'] = df['SBA_Appv'] / df['GrAppv']
    df['realstate'] = 0
    df['realstate'] = (df['Term'] > 240)
    df['recession'] = ((2007 - 2000 <= df['ApprovalFY']) & (df['ApprovalFY'] <= 2009 - 2000))
    df['nowadays'] = ((2010 - 2000 <= df['ApprovalFY']))

    # In[39]:


    # nai = {11: 'Agriculture', 21: 'Mining', 22: 'Utilities', 23: 'Construction', 31: 'Manufacturing',
    #        32: 'Manufacturing',
    #        33: 'Manufacturing', 42: 'Wholesale', 44: 'Retail', 45: 'Retail', 48: 'Transportation',
    #        49: 'Transportation', 51: 'Information', 52: 'Finance', 53: 'Real estate',
    #        54: 'Professional', 55: 'Management', 56: 'Administrative', 61: 'Educational',
    #        62: 'Health', 71: 'Arts', 72: 'Accommodation', 81: 'Other', 92: 'Public'}
    # def NAICS(a):
    #     if (int(a) // 10000 in nai): return nai[int(a) // 10000]
    #     return "no"
    # df['NAICS2'] = df.apply(lambda row: NAICS(row['NAICS']), axis=1)
    # print(np.sum(df['NAICS2'] == 'no'))
    # for i, j in nai.items():
    #     print(j, np.sum(df['NAICS2'] == j))


    for i in df.columns:
        print(i, df[i].nunique())

    # In[43]:


    df = df.drop(['Bank','Name','City','Zip','ApprovalDate','BalanceGross','Id'], 1)

    # In[44]:


    if showimage:
        sb.clustermap(df.corr(), annot=True)
        plt.show()

    # In[45]:


    df.info()

    # In[46]:

    tostr = ['FranchiseCode', 'NewExist', 'UrbanRural', 'State', 'BankState', 'RevLineCr',
             'LowDoc', 'SequBS', 'recession', 'nowadays', 'realstate']
    extra = ['NAICS2']
    for i in tostr:
        df[i] = df[i].astype(str)

    # In[47]:


    df = pd.get_dummies(df)

    print(df.info())

    return df


if __name__ == '__main__':
    dfy = pd.read_csv("Ytrain.csv")
    dfy = dfy.drop(['Id'], 1)
    dfy.columns = ['Y']
    dfx = pd.read_csv("Xtrain.csv")
    df = pd.concat([dfx, dfy], axis=1)

    df=fe(df)

    dftest = pd.read_csv("Xtest.csv")
    dftest=fe(dftest)
    X = df.loc[:, df.columns != 'Y'].values
    Y = df['Y'].values
    Xftest=dftest.values
    print('Xftest.shape',Xftest.shape)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    df = pd.read_csv("Ytrain.csv")


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


    estimators=[
        ('xgb',XGBClassifier(objective='binary:logistic', silent=True, nthread=2, seed=0, verbosity=0,
                     **{'subsample': 0.9, 'n_estimators': 700, 'min_child_weight': 3, 'max_depth': 10, 'learning_rate': 0.05, 'colsample_bytree': 0.7})),
        ('cb',cb.CatBoostClassifier(random_seed=0,silent=True,
                             **{'learning_rate': 0.15, 'l2_leaf_reg': 9, 'iterations': 300, 'depth': 7})),
        ('lgb',lgb.LGBMClassifier(random_state=0,silent = True,
                          **{'subsample_freq': 20, 'subsample': 0.9, 'reg_lambda': 1.2, 'reg_alpha': 1.1, 'num_leaves': 1200, 'n_estimators': 700, 'min_split_gain': 0.4, 'max_depth': 25, 'colsample_bytree': 0.7})),
    ]

    # In[ ]:


    votingC = VotingClassifier(estimators=estimators, voting='soft', n_jobs=4)
    # votingC = votingC.fit(x_train, y_train)
    # y_pred = votingC.predict(x_test)
    # acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    # print(acc)

    votingC = votingC.fit(X, Y)
    y_pred = votingC.predict(Xftest)
    print('y_pred.shape',y_pred.shape)
    y_pred=y_pred.reshape(-1)
    id=np.arange(y_pred.shape[0]).tolist()
    y_pred=y_pred.tolist()
    ans=pd.DataFrame(list(zip(id, y_pred)), columns=['Id', 'ChargeOff'])
    ans.to_csv("ans.csv",index=False)

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
