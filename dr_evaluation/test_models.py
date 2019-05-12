from pge_events import pge_events
from model_objects import all_models
from get_test_days import get_test_data, get_window_of_day
from sklearn.metrics import mean_squared_error

import datetime
import os
import pandas as pd
import numpy as np
import pickle
import operator


def test_models(site, models='all'):

    # initialize directory for model files
    if not os.path.exists('./models/{}'.format(site)):
        os.mkdir('./models/{}'.format(site))

    # By default, test all models
    if models == 'all':
        models = all_models.keys()

    start_train = pd.to_datetime('2016-01-01').tz_localize('US/Pacific').isoformat()
    end_train = pd.to_datetime(datetime.datetime.today().date()).tz_localize('US/Pacific').isoformat()
    
    # Use datetime.date objects for DR-event days
    dr_event_dates = [pd.to_datetime(d).date() for d in pge_events]

    # Get days that are similar to DR-event days to test the regression model on
    test_days, train_days = get_test_data(site, dr_event_dates, start_train, end_train)

    # train baseline model on days exlcuding event days and our test set
    exclude_dates = np.concatenate((test_days, dr_event_dates))

    # test baseline on days similar to event days, and save results
    model_errors = {}

    for model_name in models:

        # initialize the model
        model_class = all_models[model_name]['model_object']
        init_args = all_models[model_name]['init_args']
        if init_args is not None:
            model = model_class(init_args)
        else:
            model = model_class()
        if model_name[:5] == 'power' or model_name[:7] == 'weather':
            model.train(site, dr_event_dates)
        else:
            model.train(site, exclude_dates)
        
        # test on all test days
        errors = []

        for date in test_days:
            actual, prediction = model.predict(site, date)
            try:
                errors.append(mean_squared_error(actual, prediction))
            except Exception as e:
                print(e)

        test_rmse = np.sqrt(np.mean(errors))
        model.rmse = test_rmse
        write_file_path = './models/{}/{}.txt'.format(site, model_name)
        write_file = open(write_file_path, 'wb')
        pickle.dump(model, write_file)

        model_errors[model] = test_rmse

    print(model_errors)

    best_model = min(model_errors.items(), key=operator.itemgetter(1))[0]

    write_file_path = './models/{}/best.txt'.format(site)
    write_file = open(write_file_path, 'wb')
    pickle.dump(best_model, write_file)
    