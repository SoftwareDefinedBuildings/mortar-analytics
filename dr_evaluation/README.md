# Demand Response Event Evalutation
This pymortar application evaluates building energy consumption during Demand Response events. Using Ridge Regression on outdoor air temperature and time features, a baseline is predicted to estimate the energy consumption without an intervention. Then, this baseline is compared with the actual energy consumption to determine savings during the DR event.

## Instructions
1. Register for pymortar (more info [here](https://mortardata.org/docs/quick-start/))
2. Add `MORTAR_API_USERNAME` and `MORTAR_API_PASSWORD` to your environment variables.
3. Install the requirements from `requirements.txt`
4. Fill out `config.json`. Here are the fields:

**site:** The site you want to analyze

**time:** The window of dates for training the baseline model

**dr_evaluation_dates:** The dr event dates you would like to evaluate

**dr_event_dates:** All dr event dates (necessary to exclude from training the baseline model)

5. run `python3 app.py`
6. view output directories

**test:** Baseline predictions for test days

**DR_Events:** Summary data for each DR event and corresponding 15-minute interval data.