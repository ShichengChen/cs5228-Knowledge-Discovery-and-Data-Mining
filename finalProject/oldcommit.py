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
# In[3]:
from uszipcode import SearchEngine
warnings.filterwarnings('ignore')


showimage = False


def fe(df):
    # tostr = ['Name', 'City', 'State', 'Bank', 'BankState', 'ApprovalDate', 'ApprovalFY', 'RevLineCr',
    #          'LowDoc', 'DisbursementDate', 'DisbursementGross', 'BalanceGross', 'GrAppv', 'SBA_Appv']
    # for i in tostr:
    #     df[i] = df[i].apply(str)

    print("best before begin")
    search = SearchEngine(simple_zipcode=True)
    state = df["Zip"][(df["State"].isna())].apply(lambda x: search.by_zipcode(x).state)
    print(state)
    df["State"][(df["State"].isna())] = state
    #df["census"]=df["Zip"].apply(lambda x: search.by_zipcode(x).population).copy()
    #df["census"]=df["Zip"].apply(lambda x: search.by_zipcode(x).population_density).copy()

    df["LowDoc"][(df["LowDoc"].notna()) & (df["LowDoc"] != 'Y')] = 'N'
    df["LowDoc"][df["LowDoc"].isna()] = 'nan'
    df["RevLineCr"][(df["RevLineCr"].notna()) & (df["RevLineCr"] != 'Y')] = 'N'
    df["RevLineCr"][df["RevLineCr"].isna()] = 'nan'


    #df["DisbursementDate"][df["DisbursementDate"].isna()] = '19-Oct-20']
    disnan=df["DisbursementDate"].isna()
    df["DisbursementDate"][disnan] = df['ApprovalDate'][disnan].copy()
    df['DisbursementDate'] = df["DisbursementDate"].apply(str)

    # import datetime
    # def parsedate(x):
    #     dt = parse(str(x))
    #     d = (dt - parse('19-Oct-20')).days
    #     if (d >= 0):
    #         dt = datetime.datetime(dt.year - 100, dt.month, dt.day)
    #     return dt
    # DisbursementDate = df['DisbursementDate'].apply(lambda x: parsedate(x))
    # ApprovalDate = df['ApprovalDate'].apply(lambda x: parsedate(x))
    # df['difdays'] = (DisbursementDate - ApprovalDate).apply(lambda x: x.days).copy()



    df['DisbursementDate'] = (df['DisbursementDate'].str.split("-").str[-1]).copy()
    assert(np.sum(df["DisbursementDate"].isna())==0)
    #df['DisbursementDate'] = df['DisbursementDate'].astype(float)
    df['DisbursementDate'] = df['DisbursementDate'].astype(int)
    df['DisbursementDate'] = df['DisbursementDate'].apply(lambda x: x + 2000 if x < 25 else x + 1900)
    if showimage:
        sb.countplot(x='DisbursementDate', data=df)
        plt.show()

    df["ApprovalFY"] = df["ApprovalFY"].apply(str)
    print(df['ApprovalFY'][~df["ApprovalFY"].str.isnumeric()])
    df['ApprovalFY'] = df['ApprovalFY'].str.replace(r'\D', '')
    df['ApprovalFY']=df['ApprovalFY'].astype(int)
    print('np.sum(df[appfy].isna())',np.sum(df['ApprovalFY'].isna()))

    # print(np.sum(df["DisbursementDate"].isna()), np.sum(df["ApprovalDate"].isna()), )
    # import datetime
    # def parsedate(x):
    #     dt = parse(x)
    #     d = (dt - parse('19-Oct-20')).days
    #     if (d >= 0):
    #         dt = datetime.datetime(dt.year - 100, dt.month, dt.day)
    #     return dt

    # DisbursementDate = \
    #     df['DisbursementDate'][df["DisbursementDate"].notna()]. \
    #         apply(lambda x: parsedate(x))
    #
    # ApprovalDate = \
    #     df['ApprovalDate'][df["DisbursementDate"].notna()]. \
    #         apply(lambda x: parsedate(x))
    # difdays = (DisbursementDate - ApprovalDate).apply(lambda x: x.days)
    # mdifdays = difdays.median()
    # print('mdifdays', mdifdays)
    #
    # disDatenan = \
    #     df['ApprovalDate'][df["DisbursementDate"].isna()]. \
    #         apply(lambda x: parse(x) + datetime.timedelta(days=mdifdays))
    # df['DisDate'] = parse('19-Oct-21')
    # df['DisDate'][df["DisbursementDate"].isna()] = disDatenan
    # df['DisDate'][df["DisbursementDate"].notna()] = DisbursementDate
    # assert np.sum((df['DisDate'] - parse('19-Oct-20')).apply(lambda x: x.days) >= 0) == 0
    # df['ApprovalDate'] = df['ApprovalDate'].apply(lambda x: parsedate(x))
    # #basedate = parse("20-Sep-1968")
    # # df['Appday'] = (df['ApprovalDate'] - basedate).apply(lambda x: x.days)
    # # df['Disday'] = (df['DisDate'] - basedate).apply(lambda x: x.days)
    # #df['difday'] = df['Disday'] - df['Appday']
    # df['Appyear'] = df['ApprovalDate'].apply(lambda x: x.year)
    # df['Disyear'] = df['DisDate'].apply(lambda x: x.year)
    # df.drop(['DisbursementDate', 'DisDate', 'ApprovalFY'], axis=1, inplace=True)

    # In[16]:


    dollor = ['DisbursementGross', 'GrAppv', 'SBA_Appv']
    for i in dollor:
        df[i] = df[i].apply(str)
        df[i] = df[i].str.strip("$")
        df[i] = df[i].str.replace(",", "")
        df[i] = df[i].str.replace(".", "")
        df[i] = df[i].astype(int) // 100

    # In[17]:


    #df['dif'] = (df['GrAppv'] - df['SBA_Appv']).copy()

    # In[18]:


    # df['NAICS_Sector'] = df[df['NAICS']!=0].NAICS.apply(lambda d: d // 10 ** (int(math.log(d, 10)) - 1))
    # df.loc[df['NAICS'] == 0, 'NAICS_Sector'] = 0
    # df['NAICS_Sector'] = df['NAICS_Sector'].astype('int64')


    # In[19]:


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


    #np.sum((df["UrbanRural"] == 0) | (df["UrbanRural"] == 1) | (df["UrbanRural"] == 2))

    # In[29]:


    if showimage:
        sb.countplot(x='UrbanRural', data=df)
        plt.show()


    #print(np.sum(~df['FranchiseCode'].isin(['0', '1'])))
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
    df['SequBS'] = (df['State'] == df['BankState']).astype(int)
    df['portion'] = df['SBA_Appv'] / df['GrAppv']
    df['realstate'] = (df['Term'] > 240).astype(int)
    # df['recession'] = ((2007 <= df['Appyear']) & (df['Appyear'] <= 2009)).astype(int)
    # df['nowadays'] = ((2010 <= df['Appyear'])).astype(int)
    #df['before'] = (df['ApprovalFY']<2007).astype(int)
    df['recession'] = ((2007 <= df['ApprovalFY']) & (df['ApprovalFY'] <= 2009)).astype(int)
    df['nowadays'] = ((2010 <= df['ApprovalFY'])).astype(int)


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


    df = df.drop(['Bank','Name','City','Zip','ApprovalDate','BalanceGross','Id'], 1)

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

    var = ['CreateJob', 'RetainedJob', 'Term', 'GrAppv', 'SBA_Appv', 'NoEmp', 'DisbursementGross', 'portion']
    for i in range(8):
        df[var[i % 8]] = np.log(1 + df[var[i % 8]])


    df = pd.get_dummies(df)

    print(df.info())
    print(df.columns)
    print("use before best features")

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
    scores=[]
    times=[]

    def timer(start_time=None):
        if not start_time:
            start_time = datetime.now()
            return start_time
        elif start_time:
            thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
            tmin, tsec = divmod(temp_sec, 60)
            #print()
            times.append('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


    estimators = [
        ('1', XGBClassifier(objective='binary:logistic', silent=True, nthread=2, seed=0, verbosity=0,
                            **{'subsample': 1, 'n_estimators': 700, 'min_child_weight': 1, 'max_depth': 10, 'learning_rate': 0.03, 'colsample_bytree': 0.6})),
        ('2', cb.CatBoostClassifier(random_seed=0, silent=True, thread_count=2,
                                    **{'learning_rate': 0.15, 'l2_leaf_reg': 14, 'iterations': 900, 'depth': 6})),
        ('3', lgb.LGBMClassifier(random_state=0, silent=True, n_jobs=2,
                                 **{'subsample': 1, 'num_leaves': 100, 'n_estimators': 700, 'min_split_gain': 0.3, 'max_depth': 15, 'learning_rate': 0.05, 'colsample_bytree': 0.7})),
        ('4', XGBClassifier(objective='binary:logistic', silent=True, nthread=2, seed=1, verbosity=0,
                            **{'subsample': 1, 'n_estimators': 700, 'min_child_weight': 6, 'max_depth': 12, 'learning_rate': 0.03, 'colsample_bytree': 0.7})),
        ('5', cb.CatBoostClassifier(random_seed=1, silent=True, thread_count=2,
                                    **{'learning_rate': 0.1, 'l2_leaf_reg': 1, 'iterations': 900, 'depth': 7})),
        ('6', lgb.LGBMClassifier(random_state=1, silent=True, n_jobs=2,
                                 **{'subsample': 0.9, 'num_leaves': 120, 'n_estimators': 700, 'min_split_gain': 0.2, 'max_depth': 25, 'learning_rate': 0.05, 'colsample_bytree': 0.7})),
        ('7', XGBClassifier(objective='binary:logistic', silent=True, nthread=2, seed=2, verbosity=0,
                            **{'subsample': 0.8, 'n_estimators': 700, 'min_child_weight': 1, 'max_depth': 10, 'learning_rate': 0.05, 'colsample_bytree': 0.6})),
        ('8', cb.CatBoostClassifier(random_seed=2, silent=True, thread_count=2,
                                    **{'learning_rate': 0.1, 'l2_leaf_reg': 12, 'iterations': 1100, 'depth': 8})),
        ('9', lgb.LGBMClassifier(random_state=2, silent=True, n_jobs=2,
                                 **{'subsample': 0.9, 'num_leaves': 70, 'n_estimators': 1500, 'min_split_gain': 0.2, 'max_depth': 25, 'learning_rate': 0.05, 'colsample_bytree': 0.7})),
        ('10', XGBClassifier(objective='binary:logistic', silent=True, nthread=2, seed=3, verbosity=0,
                             **{'subsample': 1, 'n_estimators': 1000, 'min_child_weight': 1, 'max_depth': 10, 'learning_rate': 0.03, 'colsample_bytree': 0.7})),
        ('11', cb.CatBoostClassifier(random_seed=3, silent=True, thread_count=2,
                                     **{'learning_rate': 0.1, 'l2_leaf_reg': 11, 'iterations': 900, 'depth': 7})),
        ('12', lgb.LGBMClassifier(random_state=3, silent=True, n_jobs=2,
                                  **{'subsample': 0.8, 'num_leaves': 100, 'n_estimators': 800, 'min_split_gain': 0.3, 'max_depth': 90, 'learning_rate': 0.05, 'colsample_bytree': 0.7})),
        # ('13', XGBClassifier(objective='binary:logistic', verbosity=0,silent=True, nthread=2, seed=4, **{'subsample': 1, 'n_estimators': 1000, 'min_child_weight': 2, 'max_depth': 7, 'learning_rate': 0.03, 'colsample_bytree': 0.7})),
        # ('14', cb.CatBoostClassifier(silent=True, thread_count=2,random_seed=4,**{'learning_rate': 0.15, 'l2_leaf_reg': 11, 'iterations': 900, 'depth': 6})),
        # ('15', lgb.LGBMClassifier(silent=True, n_jobs=2,random_state=4,**{'subsample': 0.7, 'num_leaves': 100, 'n_estimators': 800, 'min_split_gain': 0.5, 'max_depth': 20, 'learning_rate': 0.05, 'colsample_bytree': 0.7})),
    ]

    # In[ ]:

    # layer_one_estimators = estimators[:10]
    # layer_two_estimators = estimators[10:]
    # layer_two = StackingClassifier(estimators=layer_two_estimators, final_estimator=LogisticRegression(), n_jobs=4)
    # clf = StackingClassifier(estimators=layer_one_estimators, final_estimator=layer_two, n_jobs=4)

    clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=8)
    # clf = StackingClassifier(
    #     estimators=estimators, final_estimator=LogisticRegression(),n_jobs=8,
    # )
    # clf = clf.fit(x_train, y_train)
    # y_pred = clf.predict(x_test)
    # acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    # print(acc)

    clf = clf.fit(X, Y)
    y_pred = clf.predict(Xftest)
    print('y_pred.shape',y_pred.shape)
    y_pred=y_pred.reshape(-1)
    id=np.arange(y_pred.shape[0]).tolist()
    y_pred=y_pred.tolist()
    ans=pd.DataFrame(list(zip(id, y_pred)), columns=['Id', 'ChargeOff'])
    ans.to_csv("ans.csv",index=False)