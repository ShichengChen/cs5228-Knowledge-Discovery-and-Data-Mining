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
from sklearn.ensemble import StackingClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from dateutil.parser import parse
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata
# In[3]:
from uszipcode import SearchEngine
warnings.filterwarnings('ignore')
# In[3]:


showimage = False


def fe(df):
    search = SearchEngine(simple_zipcode=True)
    state = df["Zip"][(df["State"].isna())].apply(lambda x: search.by_zipcode(x).state)
    # print(state)
    df["State"][(df["State"].isna())] = state

    print('wrong data lowDoc', np.sum(~df['LowDoc'].isin(['N', 'Y'])))
    df["LowDoc"][~df['LowDoc'].isin(['N', 'Y'])] = 'nan'
    assert (np.sum(df["LowDoc"].isna()) == 0 and np.sum(~df['LowDoc'].isin(['N', 'Y', 'nan'])) == 0)
    print('wrong data RevLineCr', np.sum(~df['RevLineCr'].isin(['N', 'Y'])))
    df["RevLineCr"][~df['RevLineCr'].isin(['N', 'Y'])] = 'nan'
    assert (np.sum(df["RevLineCr"].isna()) == 0 and np.sum(~df['RevLineCr'].isin(['N', 'Y', 'nan'])) == 0)

    disnan = df["DisbursementDate"].isna()
    df["DisbursementDate"][disnan] = df['ApprovalDate'][disnan].copy()
    df['DisbursementDate'] = df["DisbursementDate"].apply(str)

    # In[11]:

    df['DisbursementDate'] = df["DisbursementDate"].apply(str)
    df['DisbursementDate'] = (df['DisbursementDate'].str.split("-").str[-1])
    assert (np.sum(df["DisbursementDate"].isna()) == 0)
    df['DisbursementDate'] = df['DisbursementDate'].astype(float)
    df['DisbursementDate'] = df['DisbursementDate'].astype(int)

    # In[13]:

    df['DisbursementDate'] = df['DisbursementDate'].apply(lambda x: x + 2000 if x < 25 else x + 1900)

    # In[14]:

    if showimage:
        sb.countplot(x='DisbursementDate', data=df)
        plt.show()

    # In[15]:

    df["ApprovalFY"] = df["ApprovalFY"].apply(str)
    # print(df['ApprovalFY'][~df["ApprovalFY"].str.isnumeric()])
    df['ApprovalFY'] = df['ApprovalFY'].str.replace(r'\D', '')
    df['ApprovalFY'] = df['ApprovalFY'].astype(int)
    assert (np.sum(df['ApprovalFY'].isna()) == 0)

    # In[16]:

    dollor = ['DisbursementGross', 'GrAppv', 'SBA_Appv']
    for i in dollor:
        df[i] = df[i].apply(str)
        df[i] = df[i].str.strip("$")
        df[i] = df[i].str.replace(",", "")
        df[i] = df[i].str.replace(".", "")
        df[i] = df[i].astype(int) // 100


    if showimage:
        sb.clustermap(df.corr(), annot=True)
        plt.show()

    if showimage:
        sb.distplot(df["NAICS"])
        plt.show()

    if showimage:
        sb.distplot(df["Term"])
        plt.show()

    #df["NoEmp"] = np.log(1 + df["NoEmp"])
    if showimage:
        sb.distplot(df["NoEmp"])
        plt.show()

    df['NewExist'][df['NewExist'].isna()] = 0

    # In[25]:

    if showimage:
        sb.distplot(df["NewExist"])
        plt.show()

    # In[26]:

    #df["CreateJob"] = np.log(1 + df["CreateJob"])
    if showimage:
        sb.distplot(df["CreateJob"])
        plt.show()

    #df["RetainedJob"] = np.log(1 + df["RetainedJob"])
    if showimage:
        sb.distplot(np.log(1 + df["RetainedJob"]))
        plt.show()

    # np.sum((df["UrbanRural"] == 0) | (df["UrbanRural"] == 1) | (df["UrbanRural"] == 2))

    # In[29]:

    if showimage:
        sb.countplot(x='UrbanRural', data=df)
        plt.show()

    # print(np.sum(~df['FranchiseCode'].isin(['0', '1'])))
    # df['FranchiseCode'][df['FranchiseCode'] == 1] = 0
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
    df['SequBS'] = (df['State'] == df['BankState']).astype(int)
    df['portion'] = df['SBA_Appv'] / df['GrAppv']
    df['realstate'] = (df.Term>240).astype(int)
    df['recession'] = ((2007 <= df['ApprovalFY']) & (df['ApprovalFY'] <= 2009)).astype(int)
    df['nowadays'] = ((2010 <= df['ApprovalFY'])).astype(int)

    var = ['CreateJob', 'RetainedJob', 'GrAppv', 'SBA_Appv', 'NoEmp', 'DisbursementGross', 'portion','Term']
    for i in range(8):
        df[var[i]] = np.log(1 + df[var[i]])

    nai = {11: 'Agriculture', 21: 'Mining', 22: 'Utilities', 23: 'Construction', 31: 'Manufacturing',
           32: 'Manufacturing',
           33: 'Manufacturing', 42: 'Wholesale', 44: 'Retail', 45: 'Retail', 48: 'Transportation',
           49: 'Transportation', 51: 'Information', 52: 'Finance', 53: 'Real estate',
           54: 'Professional', 55: 'Management', 56: 'Administrative', 61: 'Educational',
           62: 'Health', 71: 'Arts', 72: 'Accommodation', 81: 'Other', 92: 'Public'}

    def NAICS(a):
        if (int(a) // 10000 in nai): return nai[int(a) // 10000]
        return "no"

    df['NAICS2'] = df.apply(lambda row: NAICS(row['NAICS']), axis=1)
    # print(np.sum(df['NAICS2'] == 'no'))
    # for i, j in nai.items():
    #     print(j, np.sum(df['NAICS2'] == j))
    #
    #
    # for i in df.columns:
    #     print(i, df[i].nunique())

    df = df.drop(['Bank', 'Name', 'City', 'Zip', 'ApprovalDate', 'BalanceGross', 'Id'], 1)

    # In[44]:

    if showimage:
        sb.clustermap(df.corr(), annot=True)
        plt.show()

    tostr = ['FranchiseCode', 'NewExist', 'UrbanRural', 'State', 'BankState', 'RevLineCr',
             'LowDoc', 'NAICS2']
    extra = []
    noneedforonehot = ['nowadays', 'recession', 'realstate', 'SequBS']
    for i in tostr:
        df[i] = df[i].astype(str)
    for i in tostr:
        df[i] = df[i].astype(str)

    df = pd.get_dummies(df)

    df = df.drop(['NAICS2_no','LowDoc_nan','RevLineCr_nan','BankState_nan','NewExist_0.0',
                  'BankState_nan'], 1)

    # bf=['CA','VA','NC','RI','DE','PA','OH','IL','MT','IN','VT','AR','OR','AI','TX']
    # sf=['CA','ND','AR','SC','AZ','IN','FL','NY','TX']

    print(df.columns.values)

    return df


if __name__ == '__main__':
    dfy = pd.read_csv("Ytrain.csv")
    dfy = dfy.drop(['Id'], 1)
    dfy.columns = ['Y']
    dfx = pd.read_csv("Xtrain.csv")
    df = pd.concat([dfx, dfy], axis=1)

    df = fe(df)

    dftest = pd.read_csv("Xtest.csv")
    dftest = fe(dftest)
    X = df.loc[:, df.columns != 'Y']
    Y = df['Y']
    Xftest = dftest.values
    print('Xftest.shape', Xftest.shape)
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


    estimators = [
        ('1', XGBClassifier(objective='binary:logistic', verbosity=0, silent=True, nthread=2, random_state=0,
                            **{'subsample': 0.8, 'n_estimators': 1500, 'min_child_weight': 1, 'max_depth': 7,
                               'learning_rate': 0.03, 'colsample_bytree': 0.8})),
        ('2', cb.CatBoostClassifier(silent=True, thread_count=2, random_seed=0,
                                    **{'learning_rate': 0.15, 'l2_leaf_reg': 11, 'iterations': 1100, 'depth': 6})),
        ('3', lgb.LGBMClassifier(silent=True, n_jobs=2, random_state=0,
                                 **{'subsample': 0.9, 'num_leaves': 120, 'n_estimators': 1500, 'min_split_gain': 0.4,
                                    'max_depth': 75, 'learning_rate': 0.01, 'colsample_bytree': 0.7})),
        ('4', XGBClassifier(objective='binary:logistic', verbosity=0, silent=True, nthread=2, random_state=1,
                            **{'subsample': 0.7, 'n_estimators': 700, 'min_child_weight': 1, 'max_depth': 12,
                               'learning_rate': 0.03, 'colsample_bytree': 0.6})),
        ('5', cb.CatBoostClassifier(silent=True, thread_count=2, random_seed=1,
                                    **{'learning_rate': 0.1, 'l2_leaf_reg': 1, 'iterations': 900, 'depth': 7})),
        ('6', lgb.LGBMClassifier(silent=True, n_jobs=2, random_state=1,
                                 **{'subsample': 1, 'num_leaves': 120, 'n_estimators': 1500, 'min_split_gain': 0.2,
                                    'max_depth': 90, 'learning_rate': 0.05, 'colsample_bytree': 0.7})),
        ('7', XGBClassifier(objective='binary:logistic', verbosity=0, silent=True, nthread=2, random_state=2,
                            **{'subsample': 0.8, 'n_estimators': 700, 'min_child_weight': 1, 'max_depth': 10,
                               'learning_rate': 0.05, 'colsample_bytree': 0.6})),
        ('8', cb.CatBoostClassifier(silent=True, thread_count=2, random_seed=2,
                                    **{'learning_rate': 0.1, 'l2_leaf_reg': 13, 'iterations': 1100, 'depth': 6})),
        ('9', lgb.LGBMClassifier(silent=True, n_jobs=2, random_state=2,
                                 **{'subsample': 0.9, 'num_leaves': 70, 'n_estimators': 1500, 'min_split_gain': 0.2,
                                    'max_depth': 25, 'learning_rate': 0.05, 'colsample_bytree': 0.7})),
        ('10', XGBClassifier(objective='binary:logistic', verbosity=0, silent=True, nthread=2, random_state=3,
                             **{'subsample': 1, 'n_estimators': 1000, 'min_child_weight': 1, 'max_depth': 10,
                                'learning_rate': 0.03, 'colsample_bytree': 0.7})),
        ('11', cb.CatBoostClassifier(silent=True, thread_count=2, random_seed=3,
                                     **{'learning_rate': 0.1, 'l2_leaf_reg': 6, 'iterations': 1100, 'depth': 7})),
        ('12', lgb.LGBMClassifier(silent=True, n_jobs=2, random_state=3,
                                  **{'subsample': 0.8, 'num_leaves': 120, 'n_estimators': 700, 'min_split_gain': 0.3,
                                     'max_depth': 90, 'learning_rate': 0.05, 'colsample_bytree': 0.7})),
        ('13', XGBClassifier(objective='binary:logistic', verbosity=0, silent=True, nthread=2, random_state=0,
                             **{'colsample_bytree': 0.9, 'learning_rate': 0.03, 'max_depth': 8, 'min_child_weight': 4,
                                'n_estimators': 1400})),
        ('14', cb.CatBoostClassifier(silent=True, thread_count=2, random_seed=0,
                                     **{'depth': 7, 'iterations': 1800, 'l2_leaf_reg': 1, 'learning_rate': 0.05,
                                        'rsm': 1})),
        ('15', lgb.LGBMClassifier(silent=True, n_jobs=2, random_state=0,
                                  **{'colsample_bytree': 0.9, 'learning_rate': 0.02, 'max_depth': 30,
                                     'n_estimators': 1400, 'num_leaves': 90})),
        ('16', XGBClassifier(objective='binary:logistic', verbosity=0, silent=True, nthread=2, seed=0,
                             **{'subsample': 1, 'n_estimators': 700, 'min_child_weight': 1, 'max_depth': 10,
                                'learning_rate': 0.03, 'colsample_bytree': 0.6})),
        ('17', cb.CatBoostClassifier(silent=True, thread_count=2, random_seed=0,
                                     **{'learning_rate': 0.15, 'l2_leaf_reg': 13, 'iterations': 1100, 'depth': 7})),
        ('18', lgb.LGBMClassifier(silent=True, n_jobs=2, random_state=0,
                                  **{'subsample': 0.7, 'num_leaves': 100, 'n_estimators': 1000, 'min_split_gain': 0.2,
                                     'max_depth': 90, 'learning_rate': 0.01, 'colsample_bytree': 0.7})),
        ('19', XGBClassifier(objective='binary:logistic', verbosity=0, silent=True, nthread=2, seed=1,
                             **{'subsample': 1, 'n_estimators': 700, 'min_child_weight': 6, 'max_depth': 12,
                                'learning_rate': 0.03, 'colsample_bytree': 0.7})),
        ('20', cb.CatBoostClassifier(silent=True, thread_count=2, random_seed=1,
                                     **{'learning_rate': 0.1, 'l2_leaf_reg': 11, 'iterations': 1100, 'depth': 7})),
        ('21', lgb.LGBMClassifier(silent=True, n_jobs=2, random_state=1,
                                  **{'subsample': 1, 'num_leaves': 100, 'n_estimators': 1000, 'min_split_gain': 0.2,
                                     'max_depth': 75, 'learning_rate': 0.01, 'colsample_bytree': 0.7})),
        ('22', XGBClassifier(objective='binary:logistic', verbosity=0, silent=True, nthread=2, seed=0,
                             **{'subsample': 0.8, 'n_estimators': 1500, 'min_child_weight': 1, 'max_depth': 7,
                                'learning_rate': 0.03, 'colsample_bytree': 0.8})),
        ('23', cb.CatBoostClassifier(silent=True, thread_count=2, random_seed=0,
                                     **{'learning_rate': 0.1, 'l2_leaf_reg': 9, 'iterations': 1100, 'depth': 7})),
        ('24', lgb.LGBMClassifier(silent=True, n_jobs=2, random_state=0,
                                  **{'subsample': 0.8, 'num_leaves': 100, 'n_estimators': 1500, 'min_split_gain': 0.2,
                                     'max_depth': 25, 'learning_rate': 0.01, 'colsample_bytree': 0.7})),
        ('25', XGBClassifier(objective='binary:logistic', verbosity=0, silent=True, nthread=2, seed=1,
                             **{'learning_rate': 0.15, 'l2_leaf_reg': 13, 'iterations': 1100, 'depth': 7})),
        ('26', cb.CatBoostClassifier(silent=True, thread_count=2, random_seed=1,
                                     **{'learning_rate': 0.15, 'l2_leaf_reg': 11, 'iterations': 900, 'depth': 6})),
        ('27', lgb.LGBMClassifier(silent=True, n_jobs=2, random_state=1,
                                  **{'subsample': 0.7, 'num_leaves': 100, 'n_estimators': 800, 'min_split_gain': 0.5,
                                     'max_depth': 20, 'learning_rate': 0.05, 'colsample_bytree': 0.7})),
        ('28', XGBClassifier(objective='binary:logistic', silent=True, nthread=2, seed=0, verbosity=0,
                             **{'subsample': 1, 'n_estimators': 700, 'min_child_weight': 1, 'max_depth': 10,
                                'learning_rate': 0.03, 'colsample_bytree': 0.6})),
        ('29', cb.CatBoostClassifier(random_seed=0, silent=True, thread_count=2,
                                     **{'learning_rate': 0.15, 'l2_leaf_reg': 14, 'iterations': 900, 'depth': 6})),
        ('30', lgb.LGBMClassifier(random_state=0, silent=True, n_jobs=2,
                                  **{'subsample': 1, 'num_leaves': 100, 'n_estimators': 700, 'min_split_gain': 0.3,
                                     'max_depth': 15, 'learning_rate': 0.05, 'colsample_bytree': 0.7})),
        ('31', XGBClassifier(objective='binary:logistic', silent=True, nthread=2, seed=1, verbosity=0,
                             **{'subsample': 1, 'n_estimators': 700, 'min_child_weight': 6, 'max_depth': 12,
                                'learning_rate': 0.03, 'colsample_bytree': 0.7})),
        ('32', cb.CatBoostClassifier(random_seed=1, silent=True, thread_count=2,
                                     **{'learning_rate': 0.1, 'l2_leaf_reg': 1, 'iterations': 900, 'depth': 7})),
        ('33', lgb.LGBMClassifier(random_state=1, silent=True, n_jobs=2,
                                  **{'subsample': 0.9, 'num_leaves': 120, 'n_estimators': 700, 'min_split_gain': 0.2,
                                     'max_depth': 25, 'learning_rate': 0.05, 'colsample_bytree': 0.7})),
        ('34', XGBClassifier(objective='binary:logistic', silent=True, nthread=2, seed=2, verbosity=0,
                             **{'subsample': 0.8, 'n_estimators': 700, 'min_child_weight': 1, 'max_depth': 10,
                                'learning_rate': 0.05, 'colsample_bytree': 0.6})),
        ('35', cb.CatBoostClassifier(random_seed=2, silent=True, thread_count=2,
                                     **{'learning_rate': 0.1, 'l2_leaf_reg': 12, 'iterations': 1100, 'depth': 8})),
        ('36', lgb.LGBMClassifier(random_state=2, silent=True, n_jobs=2,
                                  **{'subsample': 0.9, 'num_leaves': 70, 'n_estimators': 1500, 'min_split_gain': 0.2,
                                     'max_depth': 25, 'learning_rate': 0.05, 'colsample_bytree': 0.7})),
        ('37', XGBClassifier(objective='binary:logistic', silent=True, nthread=2, seed=3, verbosity=0,
                             **{'subsample': 1, 'n_estimators': 1000, 'min_child_weight': 1, 'max_depth': 10,
                                'learning_rate': 0.03, 'colsample_bytree': 0.7})),
        ('38', cb.CatBoostClassifier(random_seed=3, silent=True, thread_count=2,
                                     **{'learning_rate': 0.1, 'l2_leaf_reg': 11, 'iterations': 900, 'depth': 7})),
        ('39', lgb.LGBMClassifier(random_state=3, silent=True, n_jobs=2,
                                  **{'subsample': 0.8, 'num_leaves': 100, 'n_estimators': 800, 'min_split_gain': 0.3,
                                     'max_depth': 90, 'learning_rate': 0.05, 'colsample_bytree': 0.7})),
    ]

    # In[ ]:

    votingC = VotingClassifier(estimators=estimators, voting='soft', n_jobs=6)
    # votingC = votingC.fit(x_train, y_train)
    # y_pred = votingC.predict(x_test)
    # acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    # print(acc)

    votingC = votingC.fit(X, Y)
    y_pred = votingC.predict(Xftest)
    print('y_pred.shape', y_pred.shape)
    y_pred = y_pred.reshape(-1)
    id = np.arange(y_pred.shape[0]).tolist()
    y_pred = y_pred.tolist()
    ans = pd.DataFrame(list(zip(id, y_pred)), columns=['Id', 'ChargeOff'])

    # ans[dftest.Term == 0]= 1
    # ans[dftest.Term >=360 0] = 1 

    ans.to_csv("ans.csv", index=False)