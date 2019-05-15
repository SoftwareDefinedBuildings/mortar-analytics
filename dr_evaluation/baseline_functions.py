import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from pandas.tseries.offsets import CustomBusinessDay
import numpy as np
import datetime

def make_baseline(x_days, pivot, name="Temperature", freq="15min"):
    baseline=pivot[pivot.index.isin(x_days)].mean(axis=0)
    baseline_df=baseline.to_frame(name)

    return baseline_df


def create_timeseries(df, event_index):
    col=[]
    df.columns=['demand']
    for i in df.index:
        hours=int(i)//1
        minutes=(i%1)*60

        #col.append(event_day+pd.Timedelta(hours=hours, minutes=minutes))
        col.append(pd.Timestamp(event_index+' 00:00:00')+pd.Timedelta(hours=hours, minutes=minutes))

    df["Time"]=col

    adj_df=df.set_index(["Time"])
    df=adj_df[adj_df.columns[0]]

    return df

def select_demand(data): #removed _
    demand = data.filter(regex="demand")

    return demand

def create_pivot(data, freq="15min"): #removed _

    if freq=="15min": # we are using 15 minute intervals so we can accurately calculate cost
        data["date"] = data.index.date
        data["combined"]=data.index.hour+(data.index.minute*(1.0/60.0))
        data_multi=data.set_index(["date","combined"])
        data_multi=data_multi[~data_multi.index.duplicated(keep='last')]
        data_pivot = data_multi.unstack()
        # remove double index
        data_pivot.columns = data_pivot.columns.droplevel(0)

    elif freq=="1h":
        # add date and hour for new index
        data["date"] = data.index.date
        data["hour"] = data.index.hour
        data_multi=data.set_index(["date","hour"])
        data_multi=data_multi[~data_multi.index.duplicated(keep='last')]

        # create pivot
        data_pivot = data_multi.unstack()
        # remove double index
        data_pivot.columns = data_pivot.columns.droplevel(0)

    return data_pivot


def _remove_event_day(data, event_day, PDP_dates): #removes all event days specified in the _PDP list above

    try:

        #data = data[~(data.index.date == event_index.date())]
        data = data[~(data.index.date == event_day)]
        for i in PDP_dates:

            data=data[~(data.index.date == i)]
        return data

    except Exception as e:
        print(e)
        print("error in _remove_event_day")
        return data


def _remove_WE_holidays_NaN(data):

    no_WE = ~((data.index.weekday == 5) | (data.index.weekday == 6)) # remove if WE

    cal = calendar()
    start = datetime.datetime.strftime(data.index.min(),"%Y-%m-%d")
    end =datetime.datetime.strftime(data.index.max(),"%Y-%m-%d")
    hol_cal = cal.holidays(start=start, end=end)
    no_hol = ~data.index.isin(hol_cal) # remove if it is a national holiday

    no_NaN = ~data.isna().all(axis=1) # remove if has any NaN for any hour

    return data[no_WE & no_hol & no_NaN]


def _get_last_Y_days(data, event_index, Y):
    assert data.shape[0] >= Y, "not enough data for {} days".format(Y)
    try:
        start=data.index[0]
        data=data[start:event_index] #test this
        data = data.sort_index(ascending=False).iloc[0:Y,:]
        return data

    except Exception as e:
        print(e)
        print("data available only for {} days".format(data.shape[0]))

    return data


def _get_X_in_Y(data, power_data, X=None, event_start_h=14, event_end_h=18, weather_event_data=None, include_last=False, weather_mapping=False, weather_data=None, method='max', ):
    #choses the highest X days out of Y days (if weather_mapping is true, it choses the days with the highest OAT values)
    if not X:
        X=power_data.shape[0]
    cols = np.arange(event_start_h, event_end_h+include_last*1)

    if weather_mapping==True:
        if method=='proximity': #chooses x days based on how close the weather is
            rows=np.shape(weather_data)[0]
            weather_event_day=weather_event_data
            for i in range(rows-1):
                weather_event_data=weather_event_data.append(weather_event_day, ignore_index=True)

            weather_event_data=weather_event_data[cols]
            weather_event_data.index=weather_data[cols].index
            x_days=abs(weather_event_data-weather_data[cols]).sum(axis=1).sort_values(ascending=True)[0:X].index

        else:
            x_days=weather_data[cols].sum(axis=1).sort_values(ascending=False)[0:X].index

    else:
        x_days = power_data[cols].sum(axis=1).sort_values(ascending=False)[0:X].index
    return data[data.index.isin(x_days)], x_days


def _get_adj_ratio(data,
                    event_data,
                    event_start_h=14,
                    min_ratio=1.0,
                    max_ratio=1.3):

    # this is hardcoded, we may want to do it in a more flexible way
    # strategy: 4 hours before the event, take the first 3 and average them
    pre_event_period_start = event_start_h - 4

    try:
        ratio = event_data.iloc[:,(pre_event_period_start*4):(event_start_h-1)*4].mean().mean()/data.iloc[:,(pre_event_period_start*4):(event_start_h-1)*4].mean().mean()
#         print(ratio)
    except:
        ratio=1
        print('Error in calculating ratios')
#If you want to implement maximum and minimum restrictions uncomment lines below!

    if ratio < min_ratio:
        ratio=min_ratio
    if ratio > max_ratio:
        ratio=max_ratio

    if np.isnan(ratio):
        ratio=1
    return ratio

"""
if method='proximity' (and weather-mapping=true), then it chooses the X days that are closest to the weather in the event day,
if method='max' it chooses the hottest x days out of y days.
"""
def get_X_in_Y_baseline(data, weather_pivot, event_day,PDP_dates,
                        event_index,
                        X=3,
                        Y=10,
                        event_start_h=12,
                        event_end_h=18,
                        include_last=False,
                        adj_ratio=True,
                        min_ratio=1.0,
                        max_ratio=1.3,
                        sampling="quarterly", weather_mapping=False, method='max'):

    event_data= data[data.index.date == event_day]

    data = _remove_event_day(data, event_index,PDP_dates)

    data = _remove_WE_holidays_NaN(data)
    weather_event_data=weather_pivot[weather_pivot.index.date == event_day]
    weather_data=_remove_event_day(weather_pivot, event_index, PDP_dates)
    weather_data = _remove_WE_holidays_NaN(weather_data)

    data_y =_get_last_Y_days(data, event_index, Y)

    days=data_y.index
    weather_data=_get_last_Y_days(weather_data, event_index, Y)

    data_x, x_days = _get_X_in_Y(data, power_data=data_y,
                       X=X,
                       event_start_h=event_start_h,
                       event_end_h=event_end_h,
                        weather_event_data=weather_event_data,
                       include_last=include_last, weather_mapping=weather_mapping, weather_data=weather_data, method=method)

    if adj_ratio:

        ratio = _get_adj_ratio(data_x, event_data,
                               event_start_h=event_start_h,
                               min_ratio=min_ratio,
                               max_ratio=max_ratio)
    else:
        ratio = 1
    data_x = (data_x.mean()*ratio).to_frame() # baseline is the average of the days selected
    data_x.columns = ["baseline"]
    return data_x, days, event_data.T, x_days, ratio


def parse_date(date):
    date=str(date)
    yyyy=date[0:4]
    mm=date[5:7]
    dd=date[8:10]
    return(int(yyyy),int(mm),int(dd))

def calculate_rmse(demand_baseline, event_index):
    demand_pivot.fillna(method='bfill',inplace=True) # TODO find a better solution
    RMSE=np.sqrt(mean_squared_error(demand_baseline,demand_pivot[demand_pivot.index==event_index].T))
    return RMSE


def mape_vectorized_v2(a, b):
    mask = a != 0
    return (np.fabs(a - b)/a)[mask].mean()
