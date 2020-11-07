import pandas as pd
import numpy as np

df1=pd.read_csv("ans.csv")
df2=pd.read_csv("0.93727.csv")

df3=pd.read_csv("ans2.csv")
df4=pd.read_csv("0.93697.csv")

print(np.sum(df1.ChargeOff!=df2.ChargeOff))
print(np.sum(df1.ChargeOff==df2.ChargeOff))

print(np.sum(df3.ChargeOff!=df4.ChargeOff))
print(np.sum(df3.ChargeOff==df4.ChargeOff))

# 792
# 99208

# 886 #census
# 99114


# 787 #no census
# 99213


# 239 old commit with census population
# 99761

# 886 old commit with census density
# 99114

# 787 old commit without census
# 99213


#19
#99981 best feature before condensed