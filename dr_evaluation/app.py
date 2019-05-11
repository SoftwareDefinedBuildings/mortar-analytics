from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import pymortar
import logging 
import os
import json
import operator
import pickle

from get_test_days import get_test_data
from feature_engineering import get_time_of_week, get_t_cutoff_values
from utils import get_window_of_day, get_workdays, get_closest_station, mean_absolute_percentage_error
from daily_data import get_daily_data
import get_data as gd
import static_models as sm

from model_objects import WeatherModel, PowerModel, RidgeModel

mpl_logger = logging.getLogger('matplotlib') 
mpl_logger.setLevel(logging.WARNING) 

def fit(site, start_train, end_train, data, exclude_dates=[]):
    """
    Fit the regression model for a site during for the specified window
    exclude_dates is a an optional set of datetime.date objects to exclude from training
    cli: pymortar client
    """
    alphas = [0.0001, .001, 0.01, 0.05, 0.1, 0.5, 1, 10]

    # Get weekdays
    data['date'] = data.index.date
    weekdays = get_workdays(start_train, end_train)
    day_filter = [d in weekdays for d in data['date']]
    df = data[day_filter]
    
    # Exclude dates
    day_filter = [d not in exclude_dates for d in df.index.date]
    df = df[day_filter]
    
    # Get time of week
    df['time_of_week'] = [get_time_of_week(t) for t in df.index]
    decoy = pd.DataFrame({
    'time_of_week':np.arange(0, 480)
    })
    df = df.append(decoy, sort=False)
    indicators = pd.get_dummies(df['time_of_week'])
    df = df.merge(indicators, left_index=True, right_index=True)
    df = df.drop(labels=['time_of_week'], axis=1)
    df = df.iloc[:-480]

    # Get changes in weather from last 15 minutes
    df['change'] = (df['weather'] - np.roll(df['weather'], 1))
    
    # Get temperature cutoffs
    cutoffs = [40, 50, 60, 70, 80]
    arr = df['weather'].apply(lambda t: get_t_cutoff_values(t, cutoffs)).values
    a = np.array(arr.tolist())
    t_features = pd.DataFrame(a)
    t_features.columns = ['temp_cutoff_' + str(i) for i in cutoffs] + ['max_cutoff']
    t_features.index = df.index
    df = df.merge(t_features, left_index=True, right_index=True)
    
    # Remove NA rows
    df = df.dropna()
    df = df[df['power'] != 0]
    
    print('training model for %s...' % site)
    # Train model
    X_train, y_train = df.drop(['power', 'date', 'weather'], axis=1), df['power']
    model = RidgeCV(normalize=True, alphas=alphas)
    model.fit(pd.DataFrame(X_train), y_train)
    print('regularization param:', model.alpha_)

    # Train Error
    y_pred = model.predict(pd.DataFrame(X_train))
    print('train rmse (kW) for %s:' % site, np.sqrt(mean_squared_error(y_pred, y_train)) / 1000)

    return model

# TODO: Make a featurize function to work for fitting and predicting
 
def predict(model, site, event_day, cli):
    '''
    Takes a model from the "fit" function and predicts for the event day
    cli: pymortar client
    '''
    start, end = get_window_of_day(event_day)
    interval = '15min'
    agg = 'MEAN'

    # Get weather
    weather = gd.get_weather(site, start, end, agg=agg, window=interval, cli=cli)
    weather.index = weather.index.tz_localize('UTC').tz_convert('US/Pacific')
    weather = weather.interpolate()
    closest_station = get_closest_station(site)
    if closest_station is not None:
        weather = pd.DataFrame(weather[closest_station])
    else:
        weather = pd.DataFrame(weather.mean(axis=1))


    # Get power
    power = gd.get_power(site, start, end, agg=agg, window=interval, cli=cli) * 4

    # Merge
    weather_mean = pd.DataFrame(weather.mean(axis=1))
    power_sum = power.sum(axis=1)
    power_sum.index = power_sum.index.tz_localize('UTC').tz_convert('US/Pacific')
    power_sum = pd.DataFrame(power_sum)
    data = power_sum.merge(weather_mean, left_index=True, right_index=True)
    data.columns = ['power', 'weather']

    data['date'] = data.index.date
    df = data
    df.index = pd.DatetimeIndex(df.index)

    # Get time of week
    df['time_of_week'] = [get_time_of_week(t) for t in df.index]
    decoy = pd.DataFrame({
    'time_of_week':np.arange(0, 480)
    })
    df = df.append(decoy, sort=False)
    indicators = pd.get_dummies(df['time_of_week'])
    df = df.merge(indicators, left_index=True, right_index=True)
    df = df.drop(labels=['time_of_week'], axis=1)
    df = df.iloc[:-480]

    # Get changes in weather from last 15 minutes
    df['change'] = (df['weather'] - np.roll(df['weather'], 1))
    
    # Get temperature cutoffs
    cutoffs = [40, 50, 60, 70, 80]
    arr = df['weather'].apply(lambda t: get_t_cutoff_values(t, cutoffs)).values
    a = np.array(arr.tolist())
    t_features = pd.DataFrame(a)
    t_features.columns = ['temp_cutoff_' + str(i) for i in cutoffs] + ['max_cutoff']
    t_features.index = df.index
    df = df.merge(t_features, left_index=True, right_index=True)

    # Take away power from predicting data
    X_test, actual = df.drop(['power', 'date', 'weather'], axis=1), df['power']

    # Predict and find test error
    baseline = model.predict(X_test)
    baseline = pd.Series(baseline, index=actual.index)

    return actual, baseline

if __name__ == '__main__':
    from pge_events import pge_events
    
    with open('config.json') as f:
        config = json.load(f)

    # start pymortar client
    client = pymortar.Client()
    
    # Use datetime.date objects for DR-event days
    dr_event_dates = [pd.to_datetime(d).date() for d in config['dr_event_dates']]

    # Choose the site and dates window for training the baseline model
    sites = config['sites']
    start_train = config['time']['start_train']
    end_train = config['time']['end_train']

    interval = '15min'

    model_cvrmses = {}

    for site in sites:
        # Get weather and power data
        data = gd.get_df(site, start_train, end_train, client)

        # Get days that are similar to DR-event days to test the regression model on
        test_days, train_days = get_test_data(site, dr_event_dates, start_train, end_train, cli=client)

        # train baseline model on days exlcuding event days and our test set
        exclude_dates = np.concatenate((test_days, dr_event_dates))
        baseline_model = fit(site, start_train, end_train, data, exclude_dates=exclude_dates)

        # test baseline on days similar to event days, and save results
        errors = []
        mapes = []
        means = []
        model_errors = {}

        if not os.path.exists('./test'):
            os.mkdir('./test')
        outdir = './test/%s' % site
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        for date in test_days:
            actual, prediction = predict(baseline_model, site, date, cli=client)
            start, end = get_window_of_day(date)
            actual_weather = gd.get_weather(site, start, end, agg='MEAN', window=interval, cli=client)
            closest_station = get_closest_station(site)
            if closest_station is not None:
                actual_weather = actual_weather[closest_station]
            else:
                actual_weather = actual_weather.mean(axis=1)
            df = pd.DataFrame({
                'actual_power': actual,
                'prediction': prediction,
                #'actual_weather': actual_weather.values
            })
            df.to_csv(outdir + '/' + str(date) + '_ridge.csv')
            try:
                errors.append(mean_squared_error(actual, prediction))
                mapes.append(mean_absolute_percentage_error(actual, prediction))
                means.append(np.mean(actual))
            except Exception as e:
                print(e)
        
        # get the cumulative test RMSE
        test_rmse = np.sqrt(np.mean(errors))
        ridge_model = RidgeModel(baseline_model, dr_event_dates, test_rmse)
        model_errors[ridge_model] = test_rmse
        print('cvrmse', test_rmse / np.mean(means))
        model_cvrmses[site] = test_rmse / np.mean(means)

        # test power model

        # print("testing power model")
        # x_y_lst = [(3, 10), (5, 10), (10, 10)]
        # for x_y in x_y_lst:
        #     errors = []
        #     for date in test_days:
        #         test_date = pd.to_datetime(date).date()
        #         prediction, actual = sm.power_model(test_date, data, dr_event_dates, x_y[0], x_y[1])
        #         errors.append(mean_squared_error(actual, prediction))
        #     test_rmse = np.sqrt(np.mean(errors))
        #     power_model = PowerModel(x_y[0], x_y[1], dr_event_dates, test_rmse)
        #     model_errors[power_model] = test_rmse

        # test power model

        # print("testing weather model")
        # x_y_lst = [(5, 10), (10, 10), (15, 20), (20, 20)]
        # for x_y in x_y_lst:
        #     errors = []
        #     for date in test_days:
        #         test_date = pd.to_datetime(date).date()
        #         prediction, actual = sm.weather_model(test_date, data, dr_event_dates, x_y[0], x_y[1])
        #         errors.append(mean_squared_error(actual, prediction))
        #     test_rmse = np.sqrt(np.mean(errors))
        #     weather_model = WeatherModel(x_y[0], x_y[1], dr_event_dates, test_rmse)
        #     model_errors[weather_model] = test_rmse

        # get best model and record it

        # best_model = min(model_errors.items(), key=operator.itemgetter(1))[0]
        # if not os.path.exists('./models/{}'.format(site)):
        #     os.mkdir('./models/{}'.format(site))
        # write_file_path = './models/{}/best.txt'.format(site)
        # write_file = open(write_file_path, 'wb')
        # pickle.dump(best_model, write_file)
    
    pd.DataFrame(model_cvrmses, index = [0]).to_csv('cvrmse.csv')
