from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


import os
import pandas as pd
import itertools
import pickle
import random
import numpy as np
#from matplotlib import pyplot
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score, cross_validate, ParameterGrid, GridSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import itertools
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import regularizers

os.chdir('C:\\Users\\BRVAVL\\OneDrive - C&A Modas Ltda\\github_vavlopes\\greenhouse_tseries\\code')

address = [x for x in os.listdir("../../data") if x.endswith(".pickle")]

def read_pickle(adresss):
#read all available files in a function (define)
    with open("../../data/df2014-06-15.pickle",'rb') as infile:
        df = pickle.load(infile)
        df = df = df.sort_values(by=['data','hora'])
        #Holding values for results presentation
        cenario = df.cenario.unique().tolist()
        range_datas = df.range_datas.unique().tolist()
    return(df,cenario,range_datas)        

def data_preparing(df,hmlook_back,target):
    #cleaning and selecting the right columns
    vet = ['prev_'+ hm for hm in map(str,list(range(hmlook_back, (6+1), 1)))]
    vet.extend(('data','hora','medicao',
                'concat_coord', 'range_datas',
                target))
    names = "|".join(vet)
    df = df.filter(regex= names, axis=1)
    #For cleaning variables that contains target in its name (ur_prev or temp_prev)
    if hmlook_back != 1:
        vet_exclude = ['prev_'+i for i in map(str,list(range(1, (hmlook_back), 1)))]
        names_exclude = "|".join(vet_exclude)
        df = df.drop(df.filter(regex= names_exclude, axis = 1).columns, axis = 1)
        if target == 'temp':
            df = df.drop('temp_est_2', axis = 1)
    #mantaining only the complete cases
    df = df.dropna(0)
    return(df)

def Holdout_split(df):
    datas = df.data.unique()
    datas = sorted(datas)
    train = df.loc[df.data < datas[7],]
    test = df.loc[df.data >= datas[7],]
    return(train, test)

def manipulate_col(train,test):
    """ Function to apply min max scaling to numeric variables """
    #This function needs to normalize each train and test for each concat_coord
    X_train = train.drop([target,'data','hora','medicao','range_datas'], axis=1)
    X_test = test.drop([target,'data','hora','medicao','range_datas'], axis=1)
    y_train = train[target].values.reshape(-1,1)
    y_test = test[target].values.reshape(-1,1)
    return(X_train.values, X_test.values, y_train, y_test)

def create_indexcol(X_train):
    indexes = np.repeat(range(1,(5+1),1), np.ceil(X_train.shape[0]/5))
    indexes = indexes[:(X_train.shape[0])]
    #Adding column 17 for filter during Cross validation
    #X_train = np.hstack((X_train,indexes.reshape(-1,1)))
    return(indexes)

def reshape_data(X, y):

    n_features = int(X.shape[1]/(7-hmlook_back)) #calculo do numero de features
    X = X.reshape(X.shape[0],(7-hmlook_back),n_features) #samples,n_lag, n_features
    y = y.reshape(y.shape[0],)

    return(X, y)

def lstm_model(dim,time_step, n_features):
    model = Sequential()
    model.add(LSTM(100, kernel_regularizer=regularizers.l2(0.01),input_shape=(time_step, n_features)))
    model.add(Dense(1,activity_regularizer=regularizers.l1(0.01)))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return (model)

#X and y here are suposed to be train X and y
def cros_val_own(X, y,epoch,batch_size):
    indexes = create_indexcol(X)
    group_kfold = GroupKFold(n_splits=5)
    cvscores = []
    for train, test in group_kfold.split(X, y, groups = indexes):

        scaler_x = MinMaxScaler()
        scaler_x.fit(X[train])
        x_split_train = scaler_x.transform(X[train])
        x_split_test = scaler_x.transform(X[test])

        scaler_y = MinMaxScaler()
        scaler_y.fit(y[train])
        y_split_train = scaler_y.transform(y[train])

        x_split_train, y_split_train = reshape_data(x_split_train, y_split_train)
        time_step, n_features = x_split_train.shape[1],x_split_train.shape[2]
        # create model
        model = lstm_model(dim = 100,time_step = time_step, n_features = n_features)
        # Fit the model
        model.fit(x_split_train,y_split_train,epochs=epoch,batch_size=batch_size,verbose=2)
        x_split_test, y_split_test = reshape_data(x_split_test, y[test])
    	# evaluate the model
        yhat = model.predict(x_split_test,verbose=0)
        yhat = scaler_y.inverse_transform(yhat)
        yhat = yhat.reshape(yhat.shape[0],)
        mae = mean_absolute_error(y_split_test, yhat)
        print(mae)
        resultados_part = (mae,batch_size,epoch)
        cvscores.append(resultados_part)

    return(cvscores)

def holdout_lstm(X_train,X_test, y_train, y_test,batch_size, epoch):
    """ Description """
    scaler_x = MinMaxScaler()
    scaler_x.fit(X_train)
    X_train = scaler_x.transform(X_train)
    X_test = scaler_x.transform(X_test)

    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train)
    y_train = scaler_y.transform(y_train)
    #reshaping data
    X_train, y_train = reshape_data(X_train, y_train)
    time_step, n_features = X_train.shape[1],X_train.shape[2]
    model = lstm_model(dim = 100,time_step = time_step, n_features = n_features)
    model.fit(X_train,y_train,epochs=epoch,batch_size=batch_size,verbose=2)
    X_test, y_test = reshape_data(X_test, y_test)
	# evaluate the model
    yhat = model.predict(X_test,verbose=0)
    yhat = scaler_y.inverse_transform(yhat)
    yhat = yhat.reshape(yhat.shape[0],)
    mae = mean_absolute_error(y_test, yhat)
    print(mae)
    resultados = (mae)
    return(resultados, y_test, yhat)

def nestedCV_Hout(target,hmlook_back,address,grid):
    df_init,cenario,range_datas = read_pickle(address)
    target = target
    hmlook_back = hmlook_back
    dfi = data_preparing(df_init,hmlook_back,target)
    concat_coord_un = dfi.concat_coord.unique().tolist()
    res_comp = pd.DataFrame()
    for i in range(len(concat_coord_un)):
        df = dfi.loc[dfi.concat_coord == concat_coord_un[i],:]
        concat_coord = df.concat_coord.unique().tolist()
        df = df.drop(['concat_coord'], axis = 1)
        train, test = Holdout_split(df)
        data = test.data.tolist()
        hora = test.hora.tolist()
        X_train, X_test, y_train, y_test = manipulate_col(train, test)
    
        CV_scores = []
        for i in range(len(list(grid))):
            CV_scores.append(cros_val_own(X_train, y_train,epoch = list(grid)[i]['epochs'],batch_size = list(grid)[i]['batch_size']))
    
        dat = pd.DataFrame()
        for i in range(len(CV_scores)):
            dat = dat.append(pd.DataFrame(CV_scores[i],columns=['mae', 'batch_size','epoch']), ignore_index=True)
    
        #para guardar a melhor combinacao de hiperparametros
        best = dat.groupby(['batch_size','epoch']).mean().sort_values('mae')
        best = best['mae'].index[0]
        batch_size, epoch = best
    
        mae_final, yobs, ypred = holdout_lstm(X_train, X_test, y_train, y_test,batch_size, epoch)
    
        d = {'tecnica':'ann_lstm',
             'cenario': cenario * len(yobs), 'range_datas': range_datas * len(yobs),
             'concat_coord': concat_coord * len(yobs),
             'data': data, 'hora': hora,
             'yobs':yobs, 'ypred':ypred}
        res_comp = res_comp.append(pd.DataFrame(data=d))
        path_save = "../../results/lstm/" + str(cenario) + "_" + str(target) + "_" + str(hmlook_back) + ".csv"
    res_comp.to_csv(path_save)
    return(res_comp)

#definindo grid de CV
grid = [{'epochs': [10,20,100], 'batch_size': [1,10,45,100]}]
grid = ParameterGrid(grid)

#montagem da lista para iterar
target=['ur','temp']
address = address
hmlook_back = range(1,6,1)

iterator = list(itertools.product(target, hmlook_back,address))

for it in range(len(iterator)):
    target,hmlook_back,address = iterator[it]
    res_comp = nestedCV_Hout(target,hmlook_back,address,grid)
