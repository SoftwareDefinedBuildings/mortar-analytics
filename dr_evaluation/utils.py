import pandas as pd
import os, sys
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.utils import check_array
import numpy as np
from datetime import timedelta

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))+'/'

def mean_absolute_percentage_error(y_true, y_pred): 
    mask = y_true != 0
    return (np.fabs(y_true - y_pred)/y_true)[mask].mean() 

# function that returns a list of days not including weekends, holidays, or event day
# if pge == True will return weekdays for PG&E otherwise it will return weekdays for SCE
def get_workdays(start,end):
    start = pd.to_datetime(start).date()
    end = pd.to_datetime(end).date()
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    workdays = pd.DatetimeIndex(start=start, end=end, freq=us_bd)
    return workdays

# Returns the start and end timestamp of a single day
def get_window_of_day(date):
    start, end = pd.date_range(start=date, periods=2, freq='1d', tz='US/Pacific')
    start_ts = start.isoformat()
    end_ts = end.isoformat()
    return start_ts, end_ts

def get_closest_station(site):
    stations = pd.read_csv(os.path.join(PROJECT_ROOT, 'weather_stations.csv'), index_col='site')
    try:
        uuid = stations.loc[site].values[0]
        return uuid
    except:
        print("couldn't find closest weather station for %s" % site)
        return None

def get_date_str(date):
    date = pd.to_datetime(date).date()
    return format(date)

def get_month_window(date):
    end_date = pd.to_datetime(date).date() + timedelta(days=2)
    start_date = end_date - timedelta(days=30)
    start_ts = pd.to_datetime(start_date).tz_localize('US/Pacific').isoformat()
    end_ts = pd.to_datetime(end_date).tz_localize('US/Pacific').isoformat()
    return start_ts, end_ts