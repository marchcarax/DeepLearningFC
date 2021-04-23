from cnn_lstm_model import model
from data import data_process
import plot

def main():
    
    #Send data to analyze and predict
    start_date = "2018-01-01"
    end_date = "2021-04-21"
    stock = 'FB'
    data = data_process(stock, start_date, end_date)
    lstm_model = model(data)
    
    y_test = data.y_test
    y_test = data.scy.inverse_transform(y_test.reshape(-1,1))
    res = data.scy.inverse_transform(lstm_model.results)
    forecast = data.scy.inverse_transform(lstm_model.future)
    
    plot.plot_test(data.date, data.split, y_test, res)
    plot.plot_future(forecast, 5)

    '''
    next steps:
        create a train function or file, send the model and train the data --> DONE
        create a predict function or file that test the data --> DONE
        inverse transform here --> DONE
        plot the results --> DONE
        forecast next possible periods --> DONE??? create a simulation of 5 possible FC
        label prediction with custom indicator

        future:
        new mix in layers
        try with my custom indicator
        add compatibility with other systems like xgb and hmm
    '''

if __name__ == '__main__':
    main()