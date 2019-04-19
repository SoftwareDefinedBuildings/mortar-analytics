

import dataclient
import pandas as pd
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from pandas.tseries.offsets import CustomBusinessDay

import matplotlib.pyplot as plt
from matplotlib import style

import numpy as np
from numpy import trapz #only used in plot metric bars
#from Wrapper import *
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_array
from scipy import special
import baseline_functions as bf

def power_model(event_day, data, PDP_dates, X=10,Y=10): #event_day input must be in datetime.date(yyyy, mm, dd) format
    #power and weather are column names

    demand_pivot = bf.create_pivot(data[['power']])
    weather_pivot=bf.create_pivot(data[['weather']])
    baseline_temp=[]
    index_list=[]
    event_index=event_day.strftime('%Y-%m-%d')
    demand_baseline, days, event_data, x_days, ratio=bf.get_X_in_Y_baseline(demand_pivot,weather_pivot, event_day=event_day,
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

    return demand_baseline.T.values, demand_event

def weather_model(event_day,data, PDP_dates,X=10,Y=10):
    event_index=(str(event_day))[0:10]

    demand_pivot = bf.create_pivot(data[['power']])
    weather_pivot=bf.create_pivot(data[['weather']])
    demand_baseline, days, event_data, x_days, ratio=bf.get_X_in_Y_baseline(demand_pivot, weather_pivot, event_day,
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

    return demand_baseline.T.values[0], demand_event

    #PDP is just a placeholder for now


#print(weather_model(datetime.date(2017, 10, 26),df_joined,PDP))
