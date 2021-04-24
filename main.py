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
    df_predict = plot.plot_future(forecast, 5)
    plot.plot_label(df_predict)

    '''
    next steps:
        create a train function or file, send the model and train the data --> DONE
        create a predict function or file that test the data --> DONE
        inverse transform here --> DONE
        plot the results --> DONE
        forecast next possible periods --> DONE (???) 
        add X_fut --> DONE
        leave y(close) outside x and see how it performs --> DONE, NOT BAD :D
        create a simulation of 5 possible FC . It would need to be trained 5 times to work...it takes toomuch time --> NO
        add ci in test plot --> NO, not worth it
        label prediction with custom indicator with np.where =1 UP, =0 DOWN, or range --> DONE

        NOT WORKING WELL THE FC WHYYYYYYYYY --> USE WHAT YOU LEARNED IN SAVED PAGE
        change input data and output y data to be next day close! you are predicting after all
        do the label pred with full trained and pred data!! not close!
        add ci in test plot based on % from pred price

        future:
        new mix in layers
        try with my custom indicator
        add compatibility with other systems like xgb and hmm
        add new models for 1min (or 15m) eurusd and 1D rolling analysis as my excel model
    '''

if __name__ == '__main__':
    main()