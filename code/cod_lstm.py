
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


import os
import pandas as pd
import pickle
import random
import numpy as np
#from matplotlib import pyplot
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score, cross_validate, ParameterGrid, GridSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
import itertools
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import regularizers

# Variaveis globais
target = 'ur'
hmlook_back = 6

os.chdir('C:\\Users\\vinic\\Google Drive\\Mestrado\\pratical_project\\variability_part2\\greenhouse_tseries\\code')

#read all available files in a function (define)
with open("../../data/df2014-06-15.pickle",'rb') as infile:
    df = pickle.load(infile)
    df = df = df.sort_values(by=['data','hora'])
    #Holding values for results presentation
    cenario = df.cenario.unique().tolist()
    range_datas = df.range_datas.unique().tolist()

def data_preparing(df,hmlook_back,target):
    #cleaning and selecting the right columns
    vet = ['prev_'+i for i in map(str,list(range(hmlook_back, (6+1), 1)))]
    vet.extend((#'x','y','z',
                'data','hora','medicao',
                'concat_coord', 'range_datas',
                target))
    names = "|".join(vet)
    names
    df = df.filter(regex= names, axis=1)
    #For cleaning variables that contains target in its name (ur_prev or temp_prev)
    if hmlook_back != 1:
        vet_exclude = ['prev_'+i for i in map(str,list(range(1, (hmlook_back), 1)))]
        names_exclude = "|".join(vet_exclude)
        df = df.drop(df.filter(regex= names_exclude, axis = 1).columns, axis = 1)
    #mantaining only the complete cases
    df = df.dropna(0)
    return(df)

def Holdout_split(df):
    datas = df.data.unique()
    datas = sorted(datas)
    train = df.loc[df.data < datas[7],]
    test = df.loc[df.data >= datas[7],]
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
    indexes = np.repeat(range(1,(5+1),1), np.ceil(X_train.shape[0]/5))
    indexes = indexes[:(X_train.shape[0])]
    #Adding column 17 for filter during Cross validation
    #X_train = np.hstack((X_train,indexes.reshape(-1,1)))
    return(indexes)

#procedural part that needs to be upgraded
df = data_preparing(df,hmlook_back,target)
concat_coord = df.concat_coord.unique().tolist()
df = df.loc[df.concat_coord == concat_coord[0],:]
df = df.drop(['concat_coord'], axis = 1)
train, test = Holdout_split(df)
X_train, X_test, y_train, y_test = scaleNum_col(train, test)
#print(pd.DataFrame(X_train))

def reshape_data(X_train, X_test, y_train, y_test):

    n_features = int(X_train.shape[1]/(7-hmlook_back))
    X_train = X_train.reshape(X_train.shape[0],(7-hmlook_back),n_features)
    X_test = X_test.reshape(X_test.shape[0],(7-hmlook_back),n_features)
    y_train = y_train.reshape(y_train.shape[0],)
    y_test = y_test.reshape(y_test.shape[0],)

    return(X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test = reshape_data(X_train, X_test, y_train, y_test)
time_step, n_features = X_train.shape[1], X_train.shape[2]
def modeling_LSTM():
    model = Sequential()
    model.add(LSTM(100, kernel_regularizer=regularizers.l2(0.01)))#, input_shape=(time_step, n_features))) #input_shape = (time_step, number of features)
    model.add(Dense(1,activity_regularizer=regularizers.l1(0.01)))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

estimator = KerasRegressor(build_fn = modeling_LSTM, time_step = time_step, n_features = n_features, epochs = 10, batch_size = 2, verbose = 2)

#transformer = MinMaxScaler()
#regr = TransformedTargetRegressor(regressor = estimator, transformer = transformer)
indexes = create_indexcol(X_train)
group_kfold = GroupKFold(n_splits=5)
group_kfold = group_kfold.split(X_train, y_train, groups = indexes)
gkf = list(group_kfold)
grid = [{'epochs': [10,20], 'batch_size': [5,10]}]
grid = ParameterGrid(grid)
resultados = []
for i in range(len(list(grid))):
    results = cross_val_score(estimator, X_train, y_train, fit_params = list(grid)[i], cv = gkf, scoring='neg_mean_absolute_error')
    resultados_part = (np.mean(results),list(grid)[i]['batch_size'],list(grid)[i]['epochs'])
    resultados.append(resultados_part)

#svc = make_pipeline(StandardScaler(), svm.SVC(C=1.0, kernel="linear"))
#a = cross_validate(svc, x, y, cv = list(group_kfold), scoring='f1')
a = pd.DataFrame(resultados,columns=['score', 'batch_size', 'epochs'])

b = a.sort_values('score', ascending  = True).iloc[1,:]
print(b)

# group_kfold = GroupKFold(n_splits=5)
# group_kfold = group_kfold.split(X_train, y_train, groups = indexes)
# cvscores = []
# for train, test in kfold.split(X, Y):
#   # create model
#     model = Sequential()
#     model.add(LSTM(100, kernel_regularizer=regularizers.l2(0.01)))#, input_shape=(time_step, n_features))) #input_shape = (time_step, number of features)
#     model.add(Dense(1,activity_regularizer=regularizers.l1(0.01)))
#     model.compile(loss='mse', optimizer='adam', metrics=['mae'])
#
#
#     # Fit the model
# 	model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
# 	# evaluate the model
# 	scores = model.evaluate(X[test], Y[test], verbose=0)
# 	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# 	cvscores.append(scores[1] * 100)





#one possible solution
# def drop_nans(X, y=None):
#     total = X.shape[1]
#     new_thresh = total - thresh
#     df = pd.DataFrame(X)
#     df.dropna(thresh=new_thresh, inplace=True)
#     return df.values

#transformer = FunctionTransformer(drop_nans, validate=False)





#
# import tensorflow as tf
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
#
# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()
#
# import keras
# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)







#array([-0.07965646, -0.07173461, -0.06853584, -0.05444147, -0.07200295])

#     # fit network
#     # it could be good to use a batch size equal to the number of registers per "matricula" (maybe use the mean)
#     history = model.fit(X_train, y_train, epochs=1000, batch_size=1, validation_data=(X_test, y_test), verbose=2, shuffle=False)
#     # plot history
#     pyplot.plot(history.history['loss'], label='train')
#     pyplot.plot(history.history['val_loss'], label='test')
#     pyplot.legend()
#     pyplot.show()
#
#     yhat = model.predict(X_test)
#     return(yhat)
#
#
#
# X_train, X_test, y_train, y_test = reshape_data(X_train, X_test, y_train, y_test)
# yhat = modeling_LSTM(X_train, X_test, y_train, y_test)


#def mounting_results(yhat):







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
