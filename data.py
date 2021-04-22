#Prepare data to feed the Neural Network

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data as web


class data_process:

    def __init__(self, ticker, start_date, end_date):
        
        self.df = self.get_data(ticker, start_date, end_date)
        self.window = 60
        self.features = 6
        self.X, self.y, self.scx, self.scy = self.minmaxscale(self.df, self.window)
        self.split = 30
        self.X_train = self.X[:-self.split-1, :, :]
        self.X_test = self.X[-self.split-1:, :, :]
        self.y_train = self.y[:-self.split-1]
        self.y_test = self.y[-self.split-1:]


    def get_data(self, ticker, start_date, end_date):
        df = web.get_data_yahoo(ticker, start = start_date, end = end_date)
        df.drop('Adj Close', axis=1, inplace = True)
        df['std'] = df['Close'].rolling(10).std()
        df['std'].fillna(0)
        df['std'] = np.where(df["std"].isna(),0,df["std"]).astype("float")

        alpha = 0.55 #Historical percen up vs down
        prev_close = np.array(df['Close'].shift(1))
        close = np.array(df['Close'])
        prev_std = np.array(df['std'].shift(1))
        trend = []
        for i in range(len(df)):

            if close[i] > (prev_close[i] + (alpha*prev_std[i])):
                trend.append(1) #Up
            elif close[i] < (prev_close[i] - (alpha*prev_std[i])):
                trend.append(0) #Down
            else:
                trend.append(0.5) #Range
        
        df['trend'] = trend
        df['ema_trend'] = df['trend'].rolling(20).mean()
        print(df.tail())
        return df

    def minmaxscale(self, df, window):
        #Scaling data
        scx = MinMaxScaler(feature_range = (0, 1))
        scy = MinMaxScaler(feature_range = (0, 1))
        dfv = df[['High', 'Low', 'Open', 'Volume', 'std', 'Close']].values
        #dfv = df.iloc[:,1:6].values
        df_scale = pd.DataFrame(scx.fit_transform(dfv))
        X = df_scale[[0,1,2,3,4,5]]
        print(X.shape)
        X_aux = []

        for i in range(window, len(X)-1):
            X_aux.append(X.iloc[i-window:i, :])

        X = np.array(X_aux)
        print(X.shape)
        X = np.reshape(X, (X.shape[0], X.shape[1], 6))
        print(X.shape)
        #y = np.array(df_scale[5])
        y_close = df['Close'].values
        y = np.array(scy.fit_transform(y_close.reshape(-1,1)))
        return X, y, scx, scy

