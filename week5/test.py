# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as M
import seaborn as sns
import statsmodels.api as sm

# %%
data = pd.read_csv('https://raw.githubusercontent.com/AgusteFatu/burayaneyazcaz/main/data/ISLR_Default.csv')
data.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
# %%
data.info()
# %%
# To show a few observations of data and data length  
display(data.head())
print("\n{} Rows and {} columns.".format(data.shape[0],data.shape[1]))
# %%
data[['balance','income']].describe()
# %%
display(data['default'].value_counts())
display(data['student'].value_counts())
# %%
# Select feature and target
X = data.drop(['default'],axis=1)
y = data[['Index','default']]
# %%
from sklearn.preprocessing import OneHotEncoder

# Can not use OneHotEncoder for Target 
le = preprocessing.LabelEncoder()
ycoppy = y.copy()
y['default'] = le.fit_transform(y['default'])

# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')

# Encode column in a different dataframe and then add new dataframe to the old one 
enc_df = pd.DataFrame(enc.fit_transform(X[['student']]).toarray())
X = X.join(enc_df)
X.rename(columns={0: 'S_N', 1: 'S_Y'},inplace=True)

X = X.drop(['student'],axis=1)

# %%
X
# %%
