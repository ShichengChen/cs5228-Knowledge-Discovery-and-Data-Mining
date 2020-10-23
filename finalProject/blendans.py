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
import glob


df=pd.DataFrame(list(zip(np.arange(100000).tolist(), np.zeros(100000).tolist())),columns=['Id','c'])
for i in glob.glob('./ansfolder/*.csv'):
    print(i)
    cur=pd.read_csv(i)
    df['c']+=cur['ChargeOff']
df['ChargeOff']=0
df['ChargeOff'][df.c>=3]=1
df.drop('c',axis=1,inplace=True)
df.to_csv('ansave.csv',index=False)

