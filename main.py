from cnn_lstm_model import model
from data import data_process #change the data version to run
import plot

def main():
    
    #Send data to analyze and predict
    start_date = "2018-01-01"
    end_date = "2021-04-23"
    stock = 'FB'
    data = data_process(stock, start_date, end_date)
    lstm_model = model(data)
    
    #Prepare data to present results
    y_test = data.y_test
    y_test = data.scy.inverse_transform(y_test).T
    res = data.scy.inverse_transform(lstm_model.results).T
    forecast = data.scy.inverse_transform(lstm_model.future)

    #Plot results
    plot.plot_test(data.date, data.split, y_test[1], res)
    df_predict = plot.plot_future(forecast, 5)
    plot.plot_label(df_predict)


if __name__ == '__main__':
    main()