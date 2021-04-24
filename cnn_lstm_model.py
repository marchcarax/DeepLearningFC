#trying a mix of cnn + lstm timeseries model

import tensorflow as tf
from tensorflow.keras import Model, layers
import pandas as pd
import numpy as np
import data

class model():

    def __init__(self, data_process):
        
        self.data = data_process
        self.lstm_model = self.Mix_LSTM_Model(self.data.window, self.data.features, self.data.futureSteps)
        self.lstm_fit = self.fit_model(self.lstm_model, self.data.X_train, self.data.y_train)
        self.results = self.pred(self.lstm_fit, self.data.X_test)
        self.future = self.forecast(self.lstm_fit , self.data.X_fut)

    def Mix_LSTM_Model(self, window, features, futureSteps):

        model = tf.keras.Sequential()
        model.add(layers.Conv1D(input_shape=(window, features), filters=32,
                        kernel_size=2, strides=1, activation='relu', padding='same'))
        model.add(layers.Conv1D(filters=64, kernel_size=2, strides=1,
                        activation='relu', padding='same'))
        model.add(layers.LSTM(500, return_sequences=True))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(250,  return_sequences=False))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(100, kernel_initializer='uniform', activation='relu'))
        #Instead of the usual 1 unit in Dense, we use futureSteps to predict later n days
        model.add(layers.Dense(units = futureSteps, kernel_initializer='uniform', activation='relu'))

        model.compile(optimizer = 'adam', loss = 'mean_squared_error')

        return model
    
    def fit_model(self, Model, X_train, y_train):

        print('x shape in fit model ', X_train.shape)
        print('y shape in fit model ', y_train.shape)
        Model.fit(X_train, y_train, epochs = 10)
        Model.summary()

        return Model

    def pred(self, Model, X_test):
        
        y_pred = Model.predict(X_test)

        return y_pred

    def forecast(self, Model, X_fut):

        #I think finally have last 20 days in x_fut, follow now the steps
        print('x shape of the future: ', X_fut.shape)
        y_fut = Model.predict(X_fut)
        print('y shape of the future: ', y_fut.shape)

        return y_fut

