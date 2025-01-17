# data manipulation
import pandas as pd
import numpy as np
import math
# visualization
import seaborn as sb
import matplotlib.pyplot as plt
# model training
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# model evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from datetime import datetime
# classifiers
from sklearn.naive_bayes import GaussianNB # naive bayes
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.linear_model import LogisticRegression # logistic regression
from sklearn.tree import DecisionTreeClassifier # decision Tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold,RandomizedSearchCV
import lightgbm as lgb
from datetime import datetime
# ignore warnings
import warnings
from sklearn.ensemble import VotingClassifier
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from dateutil.parser import parse


# In[3]:
from uszipcode import SearchEngine

showimage = False


def fe(df):

    search = SearchEngine(simple_zipcode=True)
    state = df["Zip"][(df["State"].isna())].apply(lambda x: search.by_zipcode(x).state)
    print(state)
    df["State"][(df["State"].isna())] = state

    df["LowDoc"][(df["LowDoc"].notna()) & (df["LowDoc"] != 'Y')] = 'N'
    df["LowDoc"][df["LowDoc"].isna()] = 'nan'
    df["RevLineCr"][(df["RevLineCr"].notna()) & (df["RevLineCr"] != 'Y')] = 'N'
    df["RevLineCr"][df["RevLineCr"].isna()] = 'nan'

    print(np.sum(df["DisbursementDate"].isna()),np.sum(df["ApprovalDate"].isna()),)
    import datetime
    def parsedate(x):
        dt = parse(x)
        d = (dt - parse('19-Oct-20')).days
        if (d >= 0):
            dt = datetime.datetime(dt.year - 100, dt.month, dt.day)
        return dt
    # In[11]:

    DisbursementDate = \
        df['DisbursementDate'][df["DisbursementDate"].notna()]. \
            apply(lambda x: parsedate(x))

    ApprovalDate = \
        df['ApprovalDate'][df["DisbursementDate"].notna()]. \
            apply(lambda x: parsedate(x))
    difdays = (DisbursementDate - ApprovalDate).apply(lambda x: x.days)
    mdifdays = difdays.median()
    print('mdifdays', mdifdays)

    disDatenan = \
        df['ApprovalDate'][df["DisbursementDate"].isna()]. \
            apply(lambda x: parse(x) + datetime.timedelta(days=mdifdays))
    df['DisDate'] = parse('19-Oct-21')
    df['DisDate'][df["DisbursementDate"].isna()] = disDatenan
    df['DisDate'][df["DisbursementDate"].notna()] = DisbursementDate
    assert np.sum((df['DisDate'] - parse('19-Oct-20')).apply(lambda x: x.days) >= 0) == 0
    df['ApprovalDate'] = df['ApprovalDate'].apply(lambda x: parsedate(x))
    df['before'] = ((parse("01-Dec-2007") > df['DisDate'])).astype(int)
    df['recession'] = ((parse("01-Dec-2007") <= df['DisDate']) & \
                       (df['DisDate'] <= parse("30-Jun-2009"))).astype(int)
    # df['nowadays']=((parse("01-Sep-2010")<=df['DisDate'])).astype(int)
    df['nowadays'] = ((parse("30-Jun-2009") <= df['DisDate'])).astype(int)
    basedate = parse("20-Sep-1968")

    df['Appday'] = (df['ApprovalDate'] - basedate).apply(lambda x: x.days)

    df['Disday'] = (df['DisDate'] - basedate).apply(lambda x: x.days)
    df['difday'] = df['Disday'] - df['Appday']
    df['Appyear'] = df['ApprovalDate'].apply(lambda x: x.year)
    df.drop(['DisbursementDate', 'DisDate', 'ApprovalDate', 'ApprovalFY', 'Disday', 'Appday'], axis=1, inplace=True)



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


    if showimage:
        sb.distplot(df["NAICS"])
        plt.show()

    if showimage:
        sb.distplot(df["Term"])
        plt.show()

    df["NoEmp"] = np.log(1 + df["NoEmp"])
    if showimage:
        sb.distplot(df["NoEmp"])
        plt.show()

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


    df["RetainedJob"] = np.log(1 + df["RetainedJob"])
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


    df = df.drop(['Bank', 'Name', 'City', 'Zip', 'BalanceGross', 'Id'], 1)
    # In[44]:


    if showimage:
        sb.clustermap(df.corr(), annot=True)
        plt.show()

    tostr = ['FranchiseCode', 'NewExist', 'UrbanRural', 'RevLineCr',
             'LowDoc', 'NAICS2',
             'State', 'BankState',
             ]
    # extra = []
    # df = df.drop(extra, 1)

    noneedforonehot = ['nowadays', 'recession', 'realstate', 'SequBS']
    for i in tostr:
        df[i] = df[i].astype(str)

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
        ('1',XGBClassifier(objective='binary:logistic', silent=True, nthread=2, seed=0, verbosity=0,
                     **{'subsample': 0.9, 'n_estimators': 700, 'min_child_weight': 1, 'max_depth': 10, 'learning_rate': 0.03, 'colsample_bytree': 0.9})),
        ('2',cb.CatBoostClassifier(random_seed=0,silent=True,
                     **{'learning_rate': 0.15, 'l2_leaf_reg': 4, 'iterations': 900, 'depth': 6})),
        ('3',lgb.LGBMClassifier(random_state=0,silent = True,
                     **{'subsample': 0.9, 'num_leaves': 300, 'n_estimators': 1000, 'min_split_gain': 0.2, 'max_depth': 20, 'learning_rate': 0.05, 'colsample_bytree': 0.7})),
        ('4', XGBClassifier(objective='binary:logistic', silent=True, nthread=2, seed=1, verbosity=0,
                     **{'subsample': 1, 'n_estimators': 1000, 'min_child_weight': 3, 'max_depth': 10, 'learning_rate': 0.03, 'colsample_bytree': 0.7})),
        ('5', cb.CatBoostClassifier(random_seed=1, silent=True,
                     **{'learning_rate': 0.1, 'l2_leaf_reg': 13, 'iterations': 900, 'depth': 8})),
        ('6', lgb.LGBMClassifier(random_state=1, silent=True,
                     **{'subsample': 0.8, 'num_leaves': 300, 'n_estimators': 1500, 'min_split_gain': 0.2, 'max_depth': 25, 'learning_rate': 0.01, 'colsample_bytree': 0.7})),
    ]

    # In[ ]:


    votingC = VotingClassifier(estimators=estimators, voting='hard', n_jobs=6)
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

