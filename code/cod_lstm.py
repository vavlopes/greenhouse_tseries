import os
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

os.getcwd()
os.chdir("../../")

#Initial cleaning (encapsulate)
infile = open("./github_vavlopes/data/df2014-06-15.pickle",'rb')
df = pickle.load(infile)
infile.close()
df.head(2)
#acertando os casos que vieram corrompidos em dir_vento
df = df.where(a > -20000)
df = df.dropna(0)
X = df.filter(regex = 'x|y|z').values

#Function for data standardizing
X = X.astype(str)
Enc = LabelEncoder()
X[:,0] = Enc.fit_transform(X[:,0])
X[:,1] = Enc.fit_transform(X[:,1])
X[:,2] = Enc.fit_transform(X[:,2])
OneHot = OneHotEncoder(categorical_features = [0,1,2])
X = OneHot.fit_transform(X).toarray()
