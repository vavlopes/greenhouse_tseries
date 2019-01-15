import os
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


os.chdir('C:\\Users\\vinic\\Google Drive\\Mestrado\\pratical_project\\variability_part2\\greenhouse_tseries\\code')

#Initial cleaning (encapsulate)
with open("../../data/df2014-06-15.pickle",'rb') as infile:
    df = pickle.load(infile)

df.head(2)
#acertando os casos que vieram corrompidos em dir_vento
df = df.dropna(0)
X = df.filter(regex = 'x|y|z').values
X = X.astype(str)

def procLabel_col(c):
    """ Transforma as colunas objeto/label em one hot encoding """
    Enc = LabelEncoder()
    c = Enc.fit_transform(c)
    OneHot = OneHotEncoder()
    X_encoded = OneHot.fit_transform(c.reshape(-1,1)).toarray()
    return(X_encoded)

colunas_str = []
for col in range(X.shape[1]):
    colunas_str.append(procLabel_col(X[:,col]))

df = np.hstack((colunas_str[0],colunas_str[1],colunas_str[2]))
df











#Function for data standardizing

Enc = LabelEncoder()
X[:,0] = Enc.fit_transform(X[:,0])
X[:,1] = Enc.fit_transform(X[:,1])
X[:,2] = Enc.fit_transform(X[:,2])
OneHot = OneHotEncoder()
X_encoded = OneHot.fit_transform(X[:,1].reshape(-1,1)).toarray()
decoded = X_encoded.dot(OneHot.active_features_).astype(int)

X[:,1]

Enc.inverse_transform(decoded.any())

X[:,1]
