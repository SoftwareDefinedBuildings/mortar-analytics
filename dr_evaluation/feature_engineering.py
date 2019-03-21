import numpy as np

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
