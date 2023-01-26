import pandas as pd
import numpy as np
import os
import random
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, GRU
from tensorflow.keras.models import Sequential
from tensorflow.random import set_seed
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")

from keras.layers import BatchNormalization

EPOCHS=[100, 150]
CAMADAS=[1,2,3,4]
NEURONIOS=[256,128,64,32]
FUNCTION=["relu", "swish"]
FUNCTION_GRU=["tanh", "relu", "swish"]
SCALER = [StandardScaler, MinMaxScaler, PowerTransformer]
TIMESTEPS = [3,5]

PATH="https://raw.githubusercontent.com/LeiteJu/TCC/main/dados/input/"

SIGLAS = ['AC', 'AL', 'AM', 'AP', 'BA', 'CE',
  'DF', 'ES', 'GO', 'MA',
  'MG', 'MS', 'MT', 'PA', 'PB',
  'PE', 'PI', 'PR', 'RJ',
  'RN', 'RO', 'RR', 'RS',
  'SC', 'SE', 'SP', 'TO']

LABELS=["subestima: -90%", "subestima entre -90% e 60%", "subestima entre -60% e -30%",
        "subestima entre -30% e 10%", "entre -10% e 10%", "superestima entre 10% e 30%", 
        "superestima entre 30% e 60%", "superestima entre 60% e 90%", "superestima mais de 90%"]

N='NORTE'
NE="NORDESTE"
CO='CENTRO OESTE'
SE='SUDESTE'
S = 'SUL'

REGIOES = {
    'AC': N, 'AL': NE, 'AM' : N, 'AP' : N, 'BA' : NE, 'CE' : NE,
    'DF' : CO, 'ES' : SE, 'GO' : CO, 'MA' : NE,
    'MG' : SE, 'MS' : CO, 'MT' : CO, 'PA' : N, 'PB' : NE,
    'PE' : NE, 'PI' : NE, 'PR' : S, 'RJ' : SE,
    'RN' : NE, 'RO' : N, 'RR' : N, 'RS' : S,
    'SC' : S, 'SE' : NE, 'SP' : SE, 'TO' : N}

# calcula metricas de regressao
def score_regression_metrics(y_test, y_test_pred):

    RMSE = mean_squared_error(y_true=y_test, y_pred=y_test_pred, squared=False)
    MAE = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)
    MAPE = mean_absolute_percentage_error(y_true=y_test, y_pred=y_test_pred)
    R2 = r2_score(y_true=y_test, y_pred=y_test_pred)

    scores = {
        "rmse": RMSE,
        "mae": MAE,
        "mape": MAPE,
    }

    return scores

SEED = 41

def set_seeds (SEED=41):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def load_data():
    
    df = pd.read_csv(f"{PATH}processado.csv")
    
    x  = df.copy()
    x = x.sort_values(["data", "estados"])
    x = x.drop(["consumo", 'data'], axis=1)

    y = df.copy().sort_values(["data", "estados"])[['estados', 'data', 'consumo']]
    
    # processo de one-hot
    x = pd.get_dummies(data=x, columns=["estados"], drop_first=True)

    y = y['consumo']
    
    return x,y

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


class Scaler3DShape:
    
    def __init__(self, scaler=StandardScaler):
        self.scaler = scaler() 

    def fit_transform(self, x):
        x_new = self.scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        return x_new

    def transform(self, x):
        x_new = self.scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        return x_new

resultado = pd.DataFrame(columns=["gru", "dense", "epochs", "neuronios", "ativacao_gru", "ativacao_normal","normalizacao", "timesteps", "mae", "rmse", "mape"])


import itertools 
for epoch in EPOCHS:
    print (f"Epoch: {epoch}")
    for neuronios in itertools.combinations(NEURONIOS, 2):
        print (f"neuronios: {neuronios}")
        for f in FUNCTION_GRU:
            for s in SCALER:
                for t in TIMESTEPS:
                
                    set_seeds(41)

                    x,y = load_data()

                    timestep=t

                    df = x.copy()
                    df["consumo"] = y

                    df_train, df_test = train_test_split(df, test_size=0.15, shuffle=False)

                    x_train, y_train = split_sequences(df_train.values, timestep)
                    x_test, y_test = split_sequences(df_test.values, timestep)


                    scaler = Scaler3DShape(s)
                    x_train = scaler.fit_transform(x_train)
                    x_test = scaler.transform(x_test)

                    model = Sequential()
                    model.add(GRU(units=neuronios[0], return_sequences=True, activation=f)),
                    model.add(GRU(units=neuronios[1], activation=f)),
                    model.add(Dropout(rate=0.10))
                    model.add(Dense(units=1))
                    model.compile(
                        optimizer=keras.optimizers.Adam(learning_rate=0.001),
                        loss="mse",
                        metrics=[keras.metrics.RootMeanSquaredError(name="RMSE")])


                    history = model.fit(x_train, y_train, epochs=epoch, batch_size=32, verbose=0) 

                    y_pred = model.predict(x_test)
                    scores = score_regression_metrics(y_test, y_pred)
                    res = [2, 0, epoch, neuronios, f, np.nan, s, t,scores['mae'], scores['rmse'], scores['mape']]
                    resultado = resultado.append(dict(zip(resultado.columns, res)), ignore_index=True)
                    
                    
                    resultado.to_csv("2_camada_gru.csv", index=False)