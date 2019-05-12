# Demand Response Event Evalutation
This pymortar application evaluates building energy consumption during Demand Response events. Using Ridge Regression on outdoor air temperature and time features, a baseline is predicted to estimate the energy consumption without an intervention. Then, this baseline is compared with the actual energy consumption to determine savings during the DR event.

## Getting Started
1. Register for pymortar (more info [here](https://mortardata.org/docs/quick-start/))
2. Add `MORTAR_API_USERNAME` and `MORTAR_API_PASSWORD` to your environment variables.
3. Install the requirements from `requirements.txt`

## Library

**test_models.py:** Test all baseline models and determine which model has the lowest RMSE. Input the site for which you want to test the basline models. You can also select which models to test. If you do not specify models, all models from **model_objects.py** wil be used.

**evaluate.py:** Evaluate a DR event for a site at a specific date. Arguments are site name and date. Optional argument is baseline model. If a baseline model is not specified, the best model from **test_models.py** will be used. 