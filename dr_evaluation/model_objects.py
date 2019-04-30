import pandas as pd
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from pandas.tseries.offsets import CustomBusinessDay

import numpy as np
from numpy import trapz #only used in plot metric bars
#from Wrapper import *
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_array
from scipy import special
import baseline_functions as bf
from feature_engineering import get_time_of_week, get_t_cutoff_values
from utils import get_window_of_day, get_workdays, get_closest_station, get_month_window
from static_models import weather_model, power_model
import get_data as gd

class WeatherModel:
    
    def __init__(self, X, Y, PDP_dates, rmse=None):
        self.X = X
        self.Y = Y
        self.PDP_dates = PDP_dates
        self.rmse = rmse
        self.name = "Weather Model: {} out of {} last days".format(X, Y)

    def predict(self, site, event_day, cli):
        # Get the correct data for prediction
        start, end = get_month_window(event_day)
        data = gd.get_df(site, start, end, cli)
        prediction, actual = weather_model(event_day, data, self.PDP_dates, self.X, self.Y)
        return actual, prediction

class PowerModel:
    
    def __init__(self, X, Y, PDP_dates, rmse=None):
        self.X = X
        self.Y = Y
        self.PDP_dates = PDP_dates
        self.rmse = rmse
        self.name = "Power Model: {} out of {} last days".format(X, Y)

    def predict(self, site, event_day, cli):
        # Get the correct data for prediction
        start, end = get_month_window(event_day)
        data = gd.get_df(site, start, end, cli)
        prediction, actual = power_model(event_day, data, self.PDP_dates, self.X, self.Y)
        return actual, prediction

class RidgeModel:

    def __init__(self, model, PDP_dates, rmse=None):
        self.model = model
        self.PDP_dates = PDP_dates
        self.rmse = rmse
        self.name = "Ridge Model"

    def predict(self, site, event_day, cli):
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
        baseline = self.model.predict(X_test)
        baseline = pd.Series(baseline, index=actual.index)

        return actual, baseline