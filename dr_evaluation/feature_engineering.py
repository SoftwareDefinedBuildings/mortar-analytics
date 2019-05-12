import numpy as np
import pandas as pd

def get_time_of_week(dt):
    return int(96 * dt.dayofweek + 4 * dt.hour + dt.minute / 15)

def get_t_cutoff_values(temp, cutoffs):
    '''
    Implements algorithm from Mathieu et. al
    '''
    results = np.zeros(len(cutoffs) + 1)
    results[0] = min(cutoffs[0], temp)
    for i in range(1, len(cutoffs)):
        results[i] = max(min(temp, cutoffs[i]) - cutoffs[i-1], 0)
    if temp > cutoffs[-1]:
        results[-1] = temp - cutoffs[-1]
    return results

def create_ridge_features(df):
    '''
    Create feautures for dataframe with format:
        columns:
            - "power": power in watts
            - "weather": outdoor air temperature in degrees farenheit
        index:
            - 15 minute interval timestamps
    '''
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

    return df

