import os
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
import itertools

# Variaveis globais
target = 'ur'
hmlook_back = 6

os.chdir('C:\\Users\\vinic\\Google Drive\\Mestrado\\pratical_project\\variability_part2\\greenhouse_tseries\\code')

#read all available files in a function (define)
with open("../../data/df2014-06-15.pickle",'rb') as infile:
    df = pickle.load(infile)
    df = df = df.sort_values(by=['data','hora'])
    cenario = df.cenario.unique().tolist()

def data_preparing(df,hmlook_back,target):
    #cleaning and selecting the right columns
    #also split df into X and Y
    vet = ['prev_'+i for i in map(str,list(range(hmlook_back, (6+1), 1)))]
    vet.extend((#'x','y','z',
                'data','hora','medicao',
                'concat_coord', 'range_datas',
                target))
    names = "|".join(vet)
    df = df.filter(regex= names, axis=1)
    df = df.dropna(0)  #mantaining only the complete cases
    return(df)

def Holdout_split(df):
    datas = df.data.unique()
    datas = sorted(datas)
    train = df.loc[df.data < datas[7],:]
    test = df.loc[df.data >= datas[7],:]
    return(train, test)

def procLabel_col(c):
    """ Transforma uma determinada coluna objeto/label em one hot encoding """
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

def scaleNum_col(train,test):
    """ Function to apply min max scaling to numeric variables """
    #This function needs to normalize each train and test for each concat_coord
    scaler_x = MinMaxScaler()
    X_train = train.drop([target,'data','hora','medicao','range_datas'], axis=1)
    X_test = test.drop([target,'data','hora','medicao','range_datas'], axis=1)
    scaler_x.fit(X_train)
    X_train = scaler_x.transform(X_train)
    X_test = scaler_x.transform(X_test)
    scaler_y = MinMaxScaler()
    y_train = train[target].values.reshape(-1,1)
    y_test = test[target].values.reshape(-1,1)
    scaler_y.fit(y_train)
    y_train = scaler_y.transform(y_train)
    y_test = scaler_y.transform(y_test)
    return(X_train, X_test, y_train, y_test)

def create_indexcol(X_train):
    indexes = np.repeat(range(1,(10+1),1), np.ceil(X_train.shape[0]/10))
    indexes = indexes[:(X_train.shape[0])]
    #Adding column 17 for filter during Cross validation
    X_train = np.hstack((X_train,indexes.reshape(-1,1)))
    return(X_train)

#procedural part that needs to be upgraded
df = data_preparing(df,hmlook_back,target)
concat_coord = df.concat_coord.unique().tolist()
df = df.loc[df.concat_coord == concat_coord[0],:]
df = df.loc[:, df.columns != 'concat_coord']
train, test = Holdout_split(df)
X_train, X_test, y_train, y_test = scaleNum_col(train, test)
X_train = create_indexcol(X_train)
print(pd.DataFrame(X_train))









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
