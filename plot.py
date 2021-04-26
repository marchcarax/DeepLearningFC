import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
from math import sqrt
from sklearn.metrics import mean_squared_error
import seaborn as sns

def plot_test(date, test_size, y_test, results):
    
    #Model error to decide best predictions
    best_result = []
    best_rmse = 100 
    for i in range(len(results)):
        rmse = sqrt(mean_squared_error(y_test, results[i]))
        print('Simulation:', i)
        print('Mean square error between train model and test data is: %.2f'%(rmse))
        if rmse < best_rmse:
            best_result = results[i]
            best_rmse = rmse

    df = pd.DataFrame()
    df['Date'] = date[-test_size:]
    df.set_index('Date', inplace = True)
    df['predict'] = best_result
    df['real'] = y_test
    df.to_csv('data\\train_test_data.csv')

    #Plotting the results
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(df['predict'], color = 'green', label = 'Predicted Data')
    ax.plot(df['real'], color = 'red', label = 'Real Data')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.set_title('### Accuracy of the predictions:'+ str(100 - (100*(abs(df['real']-df['predict'])/df['real'])).mean().round(2))+'% ###')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    plt.legend()
    fig.savefig('figures\\cnn_lstm_train_test.png')
    plt.show()

def plot_future(future, n):

    #Preparing dataframe with test data + forecast
    df_predict = pd.read_csv('data\\train_test_data.csv')
    df_predict.set_index('Date', inplace = True)
    future_dates = future_date(df_predict.iloc[-n:,:])
    df = pd.DataFrame(index = future_dates)
    df = df[-n:]
    df['future'] = future.reshape(-1,1)

    df_predict = pd.concat([df_predict, df])
    df_predict["forecast"] = np.where(df_predict["predict"].isna(),df_predict["future"],df_predict["predict"]).astype("float")
    df_predict = df_predict.drop("predict",axis=1)
    df_predict = df_predict.drop("future",axis=1)
    #print(df_predict)

    #Plotting the results
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(df_predict['forecast'], color = 'green', label = 'Predicted Data')
    ax.plot(df_predict['real'], color = 'red', label = 'Real Data')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.axvline(x = len(df_predict)-n-1, color = 'b')
    ax.set_title('Price Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    plt.legend()
    fig.savefig('figures\\cnn_lstm_pred.png')
    plt.show()

    return df_predict[-n:]

def future_date(df: pd.DataFrame):
    #it creates the Future dates for the graphs
    date_ori = pd.to_datetime(df.index).tolist()
    for i in range(len(df)):
        date_ori.append(date_ori[-1] + timedelta(days = 1))
    date_ori = pd.Series(date_ori).dt.strftime(date_format = '%Y-%m-%d').tolist()
    return date_ori

def plot_label(future):

    #get prepared data with calcs
    df = pd.read_csv('data\\data.csv')
    df.set_index('Date', inplace = True)
    future = future['forecast']
    df = pd.concat([df, future])

    df["Close and pred"] = np.where(df["Close"].isna(),df[0],df["Close"]).astype("float")
    df = df.drop(0,axis=1)
    df = df.drop("Close",axis=1)
    df = df.drop("std", axis=1)
    df = df.drop("ema_trend", axis=1)
    df = df.drop("trend", axis=1)
    
    df['std'] = df['Close and pred'].rolling(10).std()
    df['std'] = np.where(df["std"].isna(),0,df["std"]).astype("float")

    alpha = 0.55 #Historical percen up vs down
    prev_close = np.array(df['Close and pred'].shift(1))
    close = np.array(df['Close and pred'])
    prev_std = np.array(df['std'].shift(1))
    trend = []
    for i in range(len(df)):

        if close[i] > (prev_close[i] + (alpha*prev_std[i])):
            trend.append(1) #Up
        elif close[i] < (prev_close[i] - ((1-alpha)*prev_std[i])):
            trend.append(0) #Down
        else:
            trend.append(0.5) #Range
        
    df['trend'] = trend
    df['ema_trend'] = df['trend'].rolling(10).mean()
    df['ema_trend'] = np.where(df["ema_trend"].isna(),0,df["ema_trend"]).astype("float")
    df.to_csv('data\\data_pred.csv')

    #Plotting the results
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(df['trend'][-90:], color = 'green', label = 'trend', alpha=0.2)
    ax.plot(df['ema_trend'][-90:], color = 'red', label = '10ema')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
    ax.axhline(y = 0.35, color = 'b', linestyle = '--')
    ax.axhline(y = 0.65, color = 'b', linestyle = '--')
    ax.set_title('Label plot')
    ax.set_xlabel('Time')
    ax.set_ylabel('trend')
    plt.legend()
    fig.savefig('figures\\cnn_lstm_labelpred.png')
    plt.show()
