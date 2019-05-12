# Demand Response Event Evalutation
This pymortar application evaluates building energy consumption during Demand Response events. Using Ridge Regression on outdoor air temperature and time features, a baseline is predicted to estimate the energy consumption without an intervention. Then, this baseline is compared with the actual energy consumption to determine savings during the DR event.

## Getting Started
1. Register for pymortar (more info [here](https://mortardata.org/docs/quick-start/))
2. Add `MORTAR_API_USERNAME` and `MORTAR_API_PASSWORD` to your environment variables.
3. Install the requirements from `requirements.txt`

## Library

### test_models

Test all baseline models and determine which model has the lowest RMSE. Input the site for which you want to test the basline models. You can also select which models to test. If you do not specify models, all models from **model_objects.py** wil be used.

#### Example

```
>>> from dr_evaluation.test_models import test_models

>>> test_models('avenal-veterans-hall')

{'weather_5_10': 2320.4181977646886,
 'weather_10_10': 2320.4181977646886,
 'weather_15_20': 2320.4181977646886,
 'weather_20_20': 2320.4181977646886,
 'power_3_10': 2951.6396153473383,
 'power_5_10': 2739.476438698023,
 'power_10_10': 2596.3406901230765,
 'ridge': 4333.5674809213615}
```

### evaluate

Evaluate a DR event for a site at a specific date. Arguments are site name and date. Optional argument is baseline model. If a baseline model is not specified, the best model from **test_models.py** will be used. 

#### Example

```
>>> from dr_evaluation.evaluate import evaluate

>>> evaluate('avenal-veterans-hall', '2018-07-27')

{'site': 'avenal-veterans-hall',
 'date': datetime.date(2018, 7, 27),
 'cost': {'actual': 87.63654160000002, 'baseline': 66.016124},
 'degree-days': {'cooling': None, 'heating': None},
 'baseline-type': 'Weather Model: 5 out of 10 last days',
 'baseline-rmse': 2320.4181977646886,
 'actual': array([ 2240.,  2240., ... , 2240.,  2240.]),
 'baseline': array([ 2240.,  2112., ... , 1536.,  1600.])}

>>> evaluate('avenal-veterans-hall', '2018-07-27', model_name='ridge')

{'site': 'avenal-veterans-hall',
 'date': datetime.date(2018, 7, 27),
 'cost': {'actual': 66.016124, 'baseline': 46.043598138671285},
 'degree-days': {'cooling': None, 'heating': None},
 'baseline-type': 'Ridge Model',
 'baseline-rmse': 4333.5674809213615,
 'actual': array([ 2240.,  2240., ...,  2240.,  2240.]),
 'baseline': array([6090., 6324. , ..., 6232., 5959.])}
```

