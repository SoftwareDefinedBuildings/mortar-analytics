

import dataclient
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

from .feature_engineering import get_time_of_week, get_t_cutoff_values
from .utils import get_window_of_day, get_workdays, get_closest_station, mean_absolute_percentage_error
from .baseline_functions import create_pivot, get_X_in_Y_baseline

def power_model(event_day, data, PDP_dates, X=10,Y=10): #event_day input must be in datetime.date(yyyy, mm, dd) format
    #power and weather are column names
    if type(PDP_dates[0]) == str:
        PDP_dates = pd.to_datetime(PDP_dates).date

    demand_pivot = create_pivot(data[['power']])
    weather_pivot= create_pivot(data[['weather']])
    baseline_temp=[]
    index_list=[]
    event_index=event_day.strftime('%Y-%m-%d')
    demand_baseline, days, event_data, x_days, ratio= get_X_in_Y_baseline(demand_pivot,weather_pivot, event_day=event_day,
                        PDP_dates=PDP_dates,
                        event_index=event_index,
                        X=X,
                        Y=Y,
                        event_start_h=14,
                        event_end_h=18,
                        adj_ratio=True,
                        min_ratio=1.0,
                        max_ratio=1.5,
                        sampling="quarterly")
    demand_event=demand_pivot[demand_pivot.index==event_index].values[0]
    
    prediction = to_indexed_series(demand_baseline.T.values[0], event_day)
    actual = to_indexed_series(demand_event, event_day)
    return actual, prediction 

def weather_model(event_day,data, PDP_dates,X=10,Y=10):
    if type(PDP_dates[0]) == str:
        PDP_dates = pd.to_datetime(PDP_dates).date

    event_index=(str(event_day))[0:10]

    demand_pivot =  create_pivot(data[['power']])
    weather_pivot=create_pivot(data[['weather']])
    demand_baseline, days, event_data, x_days, ratio= get_X_in_Y_baseline(demand_pivot, weather_pivot, event_day,
    PDP_dates=PDP_dates,event_index=event_index,
                        X=5,
                        Y=10,
                        event_start_h=14,
                        event_end_h=18,
                        adj_ratio=True,
                        min_ratio=1.0,
                        max_ratio=1.5,
                        sampling="quarterly",
                        weather_mapping=True , method='max')
    demand_event=demand_pivot[demand_pivot.index==event_index].values[0]

    prediction = to_indexed_series(demand_baseline.T.values[0], event_day)
    actual = to_indexed_series(demand_event, event_day)
    return actual, prediction 

    #PDP is just a placeholder for now

def to_indexed_series(array, date):
    index = pd.date_range(date, periods=96, freq='15min')
    result = pd.Series(array, index=index)
    return result
