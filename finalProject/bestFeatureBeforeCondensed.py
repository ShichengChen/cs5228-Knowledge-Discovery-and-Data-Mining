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
#from sklearn.ensemble import StackingClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from dateutil.parser import parse
# In[3]:
from uszipcode import SearchEngine
#from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


showimage = False


def fe(df):
    # tostr = ['Name', 'City', 'State', 'Bank', 'BankState', 'ApprovalDate', 'ApprovalFY', 'RevLineCr',
    #          'LowDoc', 'DisbursementDate', 'DisbursementGross', 'BalanceGross', 'GrAppv', 'SBA_Appv']
    # for i in tostr:
    #     df[i] = df[i].apply(str)



    print("best before remove nacis2 begin")
    search = SearchEngine(simple_zipcode=True)
    state = df["Zip"][(df["State"].isna())].apply(lambda x: search.by_zipcode(x).state)
    #print(state)
    df["State"][(df["State"].isna())] = state
    #df["census"] = df["Zip"].apply(lambda x: search.by_zipcode(x).population_density).copy()
    guess=False
    if guess:
        print("I want to guess") #guess result 0.93735
        df['RevLineCr'][df['RevLineCr'] == 'T']='Y'
        df['RevLineCr'][df['RevLineCr'] == '0']='N'

    print('wrong data lowDoc',np.sum(~df['LowDoc'].isin(['N', 'Y'])))
    df["LowDoc"][~df['LowDoc'].isin(['N', 'Y'])] = 'nan'
    assert (np.sum(df["LowDoc"].isna()) == 0 and np.sum(~df['LowDoc'].isin(['N', 'Y', 'nan'])) == 0)
    print('wrong data RevLineCr',np.sum(~df['RevLineCr'].isin(['N', 'Y'])))
    df["RevLineCr"][~df['RevLineCr'].isin(['N', 'Y'])] = 'nan'
    assert (np.sum(df["RevLineCr"].isna()) == 0 and np.sum(~df['RevLineCr'].isin(['N', 'Y', 'nan'])) == 0)

    #df = df.drop(['Bank', 'Name', 'City', 'Zip', 'BalanceGross', 'Id'], 1)

    #df["DisbursementDate"][df["DisbursementDate"].isna()] = '19-Oct-20']
    disnan=df["DisbursementDate"].isna()
    df["DisbursementDate"][disnan] = df['ApprovalDate'][disnan].copy()
    df['DisbursementDate'] = df["DisbursementDate"].apply(str)

    showdate=False
    if not showdate:
        print("show years")
        df['DisbursementDate'] = (df['DisbursementDate'].str.split("-").str[-1]).copy()
        assert(np.sum(df["DisbursementDate"].isna())==0)
        df['DisbursementDate'] = df['DisbursementDate'].astype(int)
        df['DisbursementDate'] = df['DisbursementDate'].apply(lambda x: x + 2000 if x < 25 else x + 1900)

        df["ApprovalFY"] = df["ApprovalFY"].apply(str)
        #print(df['ApprovalFY'][~df["ApprovalFY"].str.isnumeric()])
        df['ApprovalFY'] = df['ApprovalFY'].str.replace(r'\D', '')
        df['ApprovalFY']=df['ApprovalFY'].astype(int)
        assert (np.sum(df['ApprovalFY'].isna())==0)

        #df = df.drop(['ApprovalDate'], 1)
    else:
        print("show date")
        import datetime
        def parsedate(x):
            dt = parse(x)
            d = (dt - parse('19-Oct-20')).days
            if (d >= 0):
                dt = datetime.datetime(dt.year - 100, dt.month, dt.day)
            return dt

        df['ApprovalFY'] = df['ApprovalDate'].apply(lambda x: parsedate(x))
        df['DisbursementDate'] = df['DisbursementDate'].apply(lambda x: parsedate(x))
        assert np.sum((df['DisbursementDate'] - parse('19-Oct-20')).apply(lambda x: x.days) >= 0) == 0
        assert np.sum((df['ApprovalFY'] - parse('19-Oct-20')).apply(lambda x: x.days) >= 0) == 0


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

    df['NewExist'][df['NewExist'].isna()] = 0


    #print(np.sum(~df['FranchiseCode'].isin(['0', '1'])))
    #df['FranchiseCode'][df['FranchiseCode'] == 1] = 0
    df['FranchiseCode'][~df['FranchiseCode'].isin(['0', '1'])] = 2



    df['EMI'] = df['DisbursementGross'].astype(float) / (df['Term'] + 1)
    df['SequBS'] = (df['State'] == df['BankState']).astype(int)
    df['portion'] = df['SBA_Appv'] / df['GrAppv']
    # df['portion2'] = df['GrAppv'] / df['DisbursementGross']
    # df['portion3'] = df['SBA_Appv'] / df['DisbursementGross']
    df['realstate'] = (df['Term'] > 240).astype(int)
    #df['before'] = (df['ApprovalFY']<2007).astype(int)
    if not showdate:
        df['recession'] = ((2007 <= df['ApprovalFY']) & (df['ApprovalFY'] <= 2009)).astype(int)
        df['nowadays'] = ((2010 <= df['ApprovalFY'])).astype(int)
    else:
        df['recession'] = (
                    (parse("01-Dec-2007") <= df['ApprovalFY']) & (df['ApprovalFY'] <= parse("30-Jun-2009"))).astype(
            int)
        df['nowadays'] = ((parse("30-Jun-2009") <= df['ApprovalFY'])).astype(int)
        basedate = parse("20-Sep-1968")
        df['ApprovalFY'] = (df['ApprovalFY']).apply(lambda x: x.year)
        df['DisbursementDate'] = (df['DisbursementDate']).apply(lambda x: x.year)

    print("add more weight for term")
    df['fast'] = (df['Term'] == 0).astype(int)
    df['Tyear'] = (df['Term'] // 12).astype(int)

    var = ['CreateJob', 'RetainedJob', 'Term', 'GrAppv', 'SBA_Appv', 'NoEmp', 'DisbursementGross', 'portion']
    for i in range(8):
        #x=df[var[i % 8]]
        #     cur=df[var[i%8]].values.reshape(-1,1)
        #     pt = PowerTransformer()
        #     pt.fit(cur)
        #     df[var[i%8]]=pt.transform(cur).reshape(-1)
        df[var[i % 8]] = np.log(1 + df[var[i % 8]])
        # scaler = MinMaxScaler()
        # cur = df[var[i % 8]].values.reshape(-1, 1)
        # scaler.fit(cur)
        # df[var[i % 8]] = scaler.transform(cur).reshape(-1)

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
             'LowDoc']
    extra = []
    noneedforonehot = ['nowadays', 'recession', 'realstate', 'SequBS']
    for i in tostr:
        df[i] = df[i].astype(str)
    for i in tostr:
        df[i] = df[i].astype(str)


    df = pd.get_dummies(df)

    # print("drop multiple nan columns")
    # df = df.drop(['NewExist_0.0', 'FranchiseCode_2', 'RevLineCr_nan', 'LowDoc_nan', 'BankState_nan'], 1)

    print(df.info())
    print(df.columns.values)
    print("use before best features remove naics2")

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
    ans.to_csv("ans2.csv",index=False)
    print("out to ans2.csv")