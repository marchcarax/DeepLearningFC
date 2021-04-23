import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
from math import sqrt
from sklearn.metrics import mean_squared_error
import seaborn as sns

def plot_test(date, test_size, y_test, results):
    
    #Model error
    rmse = sqrt(mean_squared_error(y_test, results))
    print('Mean square error between train model and test data is: %.2f'%(rmse))

    df = pd.DataFrame()
    df['Date'] = date[-test_size-1:]
    df.set_index('Date', inplace = True)
    df['predict'] = results
    df['real'] = y_test
    df.to_csv('train_test_data.csv')

    #Plotting the results
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(df['predict'], color = 'green', label = 'Predicted Data')
    ax.plot(df['real'], color = 'red', label = 'Real Data')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.set_title('### Accuracy of the predictions:'+ str(100 - (100*(abs(df['real']-df['predict'])/df['real'])).mean().round(2))+'% ###')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    #why not working?
    sns.lineplot(data=df, x='Date', y='predict')
    plt.legend()
    #fig.savefig('figures\\cnn_lstm_train_test.png')
    plt.show()

def plot_future(future, n):

    #Preparing dataframe with test data + forecast
    df_predict = pd.read_csv('train_test_data.csv')
    df_predict.set_index('Date', inplace = True)
    future_dates = future_date(df_predict.iloc[-n:,:])
    df = pd.DataFrame(index = future_dates)
    df = df[-n:]
    df['future'] = future.reshape(-1,1)

    df_predict = pd.concat([df_predict, df])
    df_predict["forecast"] = np.where(df_predict["predict"].isna(),df_predict["future"],df_predict["predict"]).astype("float")
    df_predict = df_predict.drop("predict",axis=1)
    df_predict = df_predict.drop("future",axis=1)
    print(df_predict)

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
    fig.savefig('figures\\cnn_lstm_train_test.png')
    plt.show()

def future_date(df: pd.DataFrame):
    #it creates the Future dates for the graphs
    date_ori = pd.to_datetime(df.index).tolist()
    for i in range(len(df)):
        date_ori.append(date_ori[-1] + timedelta(days = 1))
    date_ori = pd.Series(date_ori).dt.strftime(date_format = '%Y-%m-%d').tolist()
    return date_ori