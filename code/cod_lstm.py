import os
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import itertools

os.chdir('C:\\Users\\vinic\\Google Drive\\Mestrado\\pratical_project\\variability_part2\\greenhouse_tseries\\code')

#Initial cleaning (encapsulate)
with open("../../data/df2014-06-15.pickle",'rb') as infile:
    df = pickle.load(infile)

def data_preparing(df,hmlook_back,target):
    #cleaning and selecting the right columns
    vet = ['prev_'+i for i in map(str,list(range(hmlook_back, (6+1), 1)))]
    vet.extend(('x','y','z',
                'data','hora','medicao',
                'concat_coord', 'cenario', 'range_datas',
                target))
    names = "|".join(vet)
    df = df.filter(regex= names, axis=1)
    df = df.dropna(0)  #mantaining only the complete cases
    return(df)

def procLabel_col(c):
    """ Transforma as colunas objeto/label em one hot encoding """
    Enc = LabelEncoder()
    c = Enc.fit_transform(c)
    OneHot = OneHotEncoder()
    X_encoded = OneHot.fit_transform(c.reshape(-1,1)).toarray()
    return(X_encoded)

def apply_EncLabel_col(df):
    """ Function to apply to one hot encode label variables (x,y,z) """
    X = df.filter(regex = 'x|y|z').values
    X = X.astype(str)
    colunas_str = []
    for col in range(X.shape[1]):
        colunas_str.append(procLabel_col(X[:,col]))

    #creating columns to be stacked to df.values in the end
    xyz = np.hstack((colunas_str[0],colunas_str[1],colunas_str[2]))
    return xyz

def scaleNum_col(df):
    """ Function to apply min max scaling to numeric variables """

df = data_preparing(df,6,'ur')
xyz = apply_EncLabel_col(df)
data = np.hstack((xyz,df.values))



print(pd.DataFrame(data))


#Function for data standardizing
#
# Enc = LabelEncoder()
# X[:,0] = Enc.fit_transform(X[:,0])
# X[:,1] = Enc.fit_transform(X[:,1])
# X[:,2] = Enc.fit_transform(X[:,2])
# OneHot = OneHotEncoder()
# X_encoded = OneHot.fit_transform(X[:,1].reshape(-1,1)).toarray()
# decoded = X_encoded.dot(OneHot.active_features_).astype(int)
#
# X[:,1]
#
# Enc.inverse_transform(decoded.any())
#
# X[:,1]
