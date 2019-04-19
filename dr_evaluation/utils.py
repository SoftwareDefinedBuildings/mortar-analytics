import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.utils import check_array
import numpy as np
from rfc3339 import format

def mean_absolute_percentage_error(y_true, y_pred): 
    mask = y_true != 0
    return (np.fabs(y_true - y_pred)/y_true)[mask].mean() 

# function that returns a list of days not including weekends, holidays, or event day
# if pge == True will return weekdays for PG&E otherwise it will return weekdays for SCE
def get_workdays(start,end):
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    workdays = pd.DatetimeIndex(start=start, end=end, freq=us_bd)
    return workdays

# Returns the start and end timestamp of a single day
def get_window_of_day(date):
    start, end = pd.date_range(start=date, periods=2, freq='1d', tz='US/Pacific')
    start_ts = format(start)
    end_ts = format(end)
    return start_ts, end_ts

def get_closest_station(site):
    stations = pd.read_csv('./weather_stations.csv', index_col='site')
    try:
        uuid = stations.loc[site].values[0]
        return uuid
    except:
        print("couldn't find closest weather station for %s" % site)
        return None