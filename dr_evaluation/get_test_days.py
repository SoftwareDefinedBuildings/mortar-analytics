import pandas as pd

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from pandas.tseries.offsets import CustomBusinessDay
import matplotlib.pyplot as plt

import numpy as np
import datetime

from .get_data import get_weather, get_df
from .utils import get_window_of_day

import pymortar

cli = pymortar.Client()

def _remove_PDP_days(data, PDP_list):
    
    try:
        for i in PDP_list:
            day=datetime.datetime.strptime(str(i), "%Y-%m-%d").date()
            data=data[~(data.index.date == day)]       
        return data
    
    except:
        print('error in _remove_PDP_days')
        return data
    

def _remove_WE_holidays_NaN(data, start, end):
    
    no_WE = ~((data.index.weekday == 5) | (data.index.weekday == 6)) # remove if WE

    cal = calendar() 
#     start = datetime.datetime.strftime(data.index.min(),"%Y-%m-%d")
#     end =datetime.datetime.strftime(data.index.max(),"%Y-%m-%d")

    hol_cal = cal.holidays(start=start, end=end)
    #hol_cal=pd.to_datetime(hol_cal).tz_localize("America/Los_Angeles") 
    no_hol = ~data.index.isin(hol_cal) # remove if it is a national holiday   
    no_NaN = ~data.isna().all(axis=1) # remove if has any NaN for any hour

    
    return data[no_WE & no_hol & no_NaN]

def isValidTestDay(date, site):
    start, end = get_window_of_day(date)
    data  =  get_df(site, start, end, agg='MEAN', interval='15min')
    for column in data.columns:
        col = data[column]
        if col.isna().sum() > 0.5*len(data):
            return False
        if (col == 0).sum() > 0.5*len(data):
            return False
        if len(col.unique()) < 3:
            return False
    return True

#%%
def get_test_data(site, PDP_days, start_search, end_search, cli=cli, fraction_test=0.5):
    means=[]
    maxes = []
    for day in PDP_days:
        start, end = get_window_of_day(day)
        weather_mean =  get_weather(site, start, end, agg='MEAN', window='24h', cli=cli)
        means.append(weather_mean)   
        weather_max =  get_weather(site, start, end, agg='MAX', window='24h', cli=cli)
        maxes.append(weather_max)
        
    means = pd.concat(means, sort=True)
    maxes = pd.concat(maxes, sort=True)
    mean_cutoff = means.median().mean()
    max_cutoff = maxes.median().mean()

    weather_mean_all =  get_weather(site, start_search, end_search, agg='MEAN', window='24h', cli=cli)
    weather_mean = weather_mean_all.mean(axis=1)  
    weather_max_all =  get_weather(site, start_search, end_search, agg='MAX', window='24h', cli=cli)
    weather_max = weather_max_all.mean(axis=1)
    
    weather = pd.DataFrame({'mean': weather_mean,'max': weather_max})
    
    
    weather=_remove_PDP_days(weather, PDP_days) 
    
    weather=_remove_WE_holidays_NaN(weather, start_search, end_search)
    
    above_mean_cuttoff = weather['mean'] >= mean_cutoff
    above_max_cutoff = weather['max'] >= max_cutoff
    
    above_cutoff = above_max_cutoff & above_mean_cuttoff
    qualified = above_cutoff[above_cutoff]
    valid_filter = [isValidTestDay(day, site) for day in qualified.index]
    qualified = qualified[valid_filter]
    num_testing_samples=int(np.ceil(np.size(qualified)*fraction_test))
        
    shuffled=list(qualified.sample(frac=1, random_state=47).index)
    weather_test=shuffled[0:num_testing_samples]
    weather_train=shuffled[num_testing_samples:]

    # convert to datetime.dates
    test_days = [t.date() for t in weather_test]
    train_days = [t.date() for t in weather_train]
    
    return test_days, train_days
