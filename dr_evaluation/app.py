from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import pymortar
import logging 
import os
import json

from get_test_days import get_test_data
from feature_engineering import get_time_of_week, get_t_cutoff_values
from utils import get_window_of_day, get_workdays, get_closest_station, mean_absolute_percentage_error
from daily_data import get_daily_data
import get_data as gd

mpl_logger = logging.getLogger('matplotlib') 
mpl_logger.setLevel(logging.WARNING) 

def fit(site, start_train, end_train, cli, exclude_dates=[]):
    """
    Fit the regression model for a site during for the specified window
    exclude_dates is a an optional set of datetime.date objects to exclude from training
    cli: pymortar client
    """
    start = start_train
    end = end_train
    interval = '15min'
    agg = 'MEAN'
    alphas = [0.0001, .001, 0.01, 0.05, 0.1, 0.5, 1, 10]

    # Get weather
    weather = gd.get_weather(site, start, end, agg=agg, window=interval, cli=cli)
    weather.index = weather.index.tz_localize('UTC').tz_convert('US/Pacific')
    closest_station = get_closest_station(site)
    if closest_station is not None:
        weather = pd.DataFrame(weather[closest_station])
    else:
        weather = pd.DataFrame(weather.mean(axis=1))

    # Get power
    power = gd.get_power(site, start, end, agg=agg, window=interval, cli=cli) * 4
    power.index = power.index.tz_localize('UTC').tz_convert('US/Pacific')

    # Merge
    weather_mean = pd.DataFrame(weather.mean(axis=1))
    power_sum = pd.DataFrame(power.sum(axis=1))
    data = power_sum.merge(weather_mean, left_index=True, right_index=True)
    data.columns = ['power', 'weather']

    # Get weekdays
    data['date'] = data.index.date
    weekdays = get_workdays(start, end)
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
    pd.DataFrame({
        'feature': X_train.columns,
        'coef': model.coef_
    }).to_csv('./model.csv')
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
    X_test.to_csv('./test_df.csv')

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

    for site in sites:
        # Get days that are similar to DR-event days to test the regression model on
        test_days, train_days = get_test_data(site, dr_event_dates, start_train, end_train, cli=client)

        # train baseline model on days exlcuding event days and our test set
        exclude_dates = np.concatenate((test_days, dr_event_dates))
        baseline_model = fit(site, start_train, end_train, exclude_dates=exclude_dates, cli=client)

        # test baseline on days similar to event days, and save results
        errors = []
        mapes = []

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
                'actual_weather': actual_weather.values
            })
            df.to_csv(outdir + '/' + str(date) + '.csv')
            try:
                errors.append(mean_squared_error(actual, prediction))
                mapes.append(mean_absolute_percentage_error(actual, prediction))
            except Exception as e:
                print(e)
        
        # get the cumulative test RMSE
        test_rmse = np.sqrt(np.mean(errors))
        print('test rmse (kW) for %s:' % site, test_rmse / 1000)

        # get the cumulative MAPE
        test_mape = np.mean(mapes)
        print('test MAPE for %s:' % site, test_mape)

        # evaluate the 10 most recent DR events for the site, and save the results
        dr_dates = [pd.to_datetime(d).date() for d in config['dr_evaluation_dates']]
        table = []
        if not os.path.exists('./DR_events'):
            os.mkdir('./DR_events')
        outdir = './DR_events/%s' % site
        if not os.path.exists('./DR_events/%s' % site):
            os.mkdir(outdir)
        for date in dr_dates:
            actual, prediction = predict(baseline_model, site, date, cli=client)
            actual_weather = gd.get_weather(site, start, end, agg='MEAN', window=interval, cli=client)
            closest_station = get_closest_station(site)
            if closest_station is not None:
                actual_weather = actual_weather[closest_station]
            else:
                actual_weather = actual_weather.mean(axis=1)
            df = pd.DataFrame({
                'actual_power': actual,
                'prediction': prediction,
                'actual_weather': actual_weather.values
            })
            df.to_csv(outdir + '/' + str(date) + '.csv')
            daily_data = get_daily_data(site, actual, prediction)
            table.append(daily_data)
        df = pd.DataFrame(table)
        df['test rmse (kw)'] = test_rmse / 1000
        df['test MAPE'] = test_mape
        df.to_csv('./DR_events/%s.csv' % site )
