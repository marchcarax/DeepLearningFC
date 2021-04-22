#trying a mix of cnn + lstm timeseries model

# Import TensorFlow v2.
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import data

class model():

    def __init__(self, ticker, start_date, end_date):
        self.data = data.data_process(ticker, start_date, end_date)
        self.lstm_model = self.Mix_LSTM_Model(self.data.window, self.data.features)

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