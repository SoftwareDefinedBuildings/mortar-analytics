import pymortar
import pandas as pd
import pickle
from .utils import get_date_str
from .daily_data import get_daily_data

def evaluate(site, date, model_name='best'):
    cli = pymortar.Client()
    date = pd.to_datetime(date).date()
    best_model_path = './models/{}/{}.txt'.format(site, model_name)
    model_file = open(best_model_path, 'rb')
    best_model = pickle.load(model_file)
    actual, prediction = best_model.predict(site, date)
    daily_data = get_daily_data(site, actual, prediction)
    return {
        'site': site,
        'date': date,
        'cost': {
            'actual': daily_data['actual_cost'],
            'baseline': daily_data['baseline_cost']
        },
        'degree-days': {
            'cooling': None,
            'heating': None
        },
        'baseline-type': best_model.name,
        'baseline-rmse': best_model.rmse,
        'actual': actual.values,
        'baseline': prediction.values
    }

def to_indexed_series(array, date):
    index = pd.date_range(date, periods=96, freq='15min')
    result = pd.Series(array, index=index)
    return result