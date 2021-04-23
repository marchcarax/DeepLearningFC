#trying a mix of cnn + lstm timeseries model

# Import TensorFlow v2.
import tensorflow as tf
from tensorflow.keras import Model, layers
import pandas as pd
import numpy as np
import data

class model():

    def __init__(self, data_process):
        #self.data = data.data_process(ticker, start_date, end_date)
        self.data = data_process
        self.lstm_model = self.Mix_LSTM_Model(self.data.window, self.data.features)
        self.lstm_fit = self.fit_model(self.lstm_model, self.data.X_train, self.data.y_train)
        self.results = self.pred(self.lstm_fit, self.data.X_test)
        self.future = self.forecast(self.lstm_fit , self.data.X_test)

    def Mix_LSTM_Model(self, window, features):

        model = tf.keras.Sequential()
        model.add(layers.Conv1D(input_shape=(window, features), filters=32,
                        kernel_size=2, strides=1, activation='relu', padding='same'))
        model.add(layers.Conv1D(filters=64, kernel_size=2, strides=1,
                        activation='relu', padding='same'))
        model.add(layers.LSTM(300, return_sequences=True))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(200,  return_sequences=False))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(100, kernel_initializer='uniform', activation='relu'))
        model.add(layers.Dense(1, kernel_initializer='uniform', activation='relu'))

        model.compile(optimizer = 'adam', loss = 'mean_squared_error')

        return model
    
    def fit_model(self, Model, X_train, y_train):

        print('x shape in fit model ', X_train.shape)
        print('y shape in fit model ', y_train.shape)
        Model.fit(X_train, y_train, epochs = 100)
        Model.summary()

        return Model

    def pred(self, Model, X_test):
        
        y_pred = Model.predict(X_test)

        return y_pred

    def forecast(self, Model, X_test):
        
        #Creates prediction given a pred_len
        pred_seq = []
        pred_len = 5
        predicted = []

        #Last n prices
        current = X_test[len(X_test)-1]

        for i in range(0, pred_len):
            predicted.append(Model.predict(current[None, :, :])[0,0])
            current = current[1:]
            #adds the new element (predictde value) at the end of the array
            current = np.insert(current, len(current), predicted[-1], axis=0)

        pred_seq.append(predicted)

        return pred_seq

