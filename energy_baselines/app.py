__author__ = "Pranav Gupta"
__email__ = "pranavhgupta@lbl.gov"

import os
import json
import pymortar
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

""" This app creates baseline of Green Button Meter data. 

Available aggregations,
pymortar.MEAN, pymortar.MAX, pymortar.MIN, 
pymortar.COUNT, pymortar.SUM, pymortar.RAW (the temporal window parameter is ignored)

Note
----
Set $MORTAR_API_USERNAME and $MORTAR_API_PASSWORD as environment variables.

"""

with open('config.json') as f:
	config = json.load(f)


def map_uuid_sitename(data):
	""" Pymortar gives back dataframe with uuid's as column names. This function maps this to its sitenames.

	Parameters
	----------
	data 	: pymoratar.result.Result
		Object returned by client.fetch(request)

	Returns
	-------
	defaultdict(list), defaultdict(list)
		map of uuid to meter data, map of uuid to oat data

	"""

	# (url, uuid, sitename)
	resp_meter = data.query('select * from view_meter')
	resp_oat = data.query('select * from view_oat')

	map_uuid_meter, map_uuid_oat = defaultdict(list), defaultdict(list)

	for (url, uuid, sitename) in resp_meter:
		map_uuid_meter[uuid].append(sitename)
	for (url, uuid, sitename) in resp_oat:
		map_uuid_oat[uuid].append(sitename)

	return map_uuid_meter, map_uuid_oat


def read_config():
	""" Reads config.json file that contains parameters for baselines and fetches data from Mortar. 

	Returns
	-------
	pd.DataFrame(), pd.DataFrame(), default(list), default(list)
		meter data, oat data, map of uuid to meter data, map of uuid to oat data
	
	"""

	# Instatiate Client
	client = pymortar.Client({})

	# Query for meter data
	query_meter = "SELECT ?meter WHERE { ?meter rdf:type/rdfs:subClassOf* brick:Green_Button_Meter };"

	# Query for outdoor air temperature data
	query_oat = """ SELECT ?t WHERE { ?t rdf:type/rdfs:subClassOf* brick:Weather_Temperature_Sensor };"""

	# Get list of sites for meter data and OAT data
	resp_meter = client.qualify([query_meter])
	resp_oat = client.qualify([query_oat])

	if resp_meter.error or resp_oat.error:
		print("ERORR: ", resp_meter.error if True else resp_oat.error)
		os._exit(0)
	else:
		# Get list of sites that are common for meter data and OAT data
		common_sites = list(set(resp_meter.sites).intersection(set(resp_oat.sites)))

		# If config['sites'] = "", then default to all sites
		if not config['sites']:
			config['sites'] = common_sites
		else:
			for site in config['sites']:
				if site not in common_sites:
					print('Incorrect site name.')
					os._exit(0)
			print("Running on {0} sites".format(len(config['sites'])))
	
	# Define the view of meters (metadata)
	meter = pymortar.View(
		name="view_meter",
		sites=config['sites'],
		definition=query_meter,
	)

	# Define the view of OAT (metadata)
	oat = pymortar.View(
		name="view_oat",
		sites=config['sites'],
		definition=query_oat
	)

	# Define the meter timeseries stream
	data_view_meter = pymortar.DataFrame(
		name="data_meter", # dataframe column name
		aggregation=pymortar.MEAN,
		window="15m",
		timeseries=[
			pymortar.Timeseries(
			view="view_meter",
			dataVars=["?meter"]
			)
		]
	)

	# Define the OAT timeseries stream
	data_view_oat = pymortar.DataFrame(
		name="data_oat",
		aggregation=pymortar.MEAN,
		window="15m",
		timeseries=[
			pymortar.Timeseries(
			view="view_oat",
			dataVars=["?t"]
			)
		]
	)

	# Define timeframe
	time_params = pymortar.TimeParams(
		start=config['time']['start'],
		end=config['time']['end']
	)

	# Form the full request object
	request = pymortar.FetchRequest(
		sites=config['sites'],
		views=[meter, oat],
		dataFrames=[data_view_meter, data_view_oat],
		time=time_params
	)

	# Fetch data from request
	data = client.fetch(request)

	# Renames columns from uuids' to sitenames'
	map_uuid_meter, map_uuid_oat = map_uuid_sitename(data)

	# Save data to csv file
	if config['save_data']:
		data['data_meter'].to_csv('meter_data.csv')
		data['data_oat'].to_csv('oat_data.csv')

	return data['data_meter'], data['data_oat'], map_uuid_meter, map_uuid_oat


def preprocess_data(data):
	""" Preprocess the data before calculating baselines.

	Parameters
	----------
	data 	: pd.DataFrame()
		data to preprocess

	Returns
	-------
	pd.DataFrame()
		processed data
	
	"""

	var_to_expand = []
	if config['time_features']['day_of_week']:
		data['dow'] = data.index.weekday
		var_to_expand.append('dow')
	if config['time_features']['time_of_day']:
		data['tod'] = data.index.hour
		var_to_expand.append('tod')

	# One-hot encode the time features
	for var in var_to_expand:
		
		add_var = pd.get_dummies(data[var], prefix=var, drop_first=True)
		
		# Add all the columns to the dataframe
		data = data.join(add_var)

		# Drop the original column that was expanded
		data.drop(columns=[var], inplace=True)

	return data


def model_data(data):
	""" Run models (3-fold cross-validation) on the processed data & compute its predictive accuracy 
	and energy savings.

	Parameters
	----------
	data 	: pd.DataFrame()
		site data to model

	Returns
	-------
	dict
		Dictionary containing adjusted r2 value and energy savings (absolute & percent)
	
	"""

	def adj_r2(r2, n, k):
		""" Calculate and return adjusted r2 score.

		Parameters
		----------
		r2  :	float
		    Original r2 score.
		n   :	int
		    Number of points in data sample.
		k   :	int
		    Number of variables in model, excluding the constant.

		Returns
		-------
		float
		    Adjusted R2 score.

		"""
		return 1 - (((1 - r2) * (n - 1)) / (n - k - 1))

	# CHECK: print data after .dropna() to see if it's not empty
	data.dropna(inplace=True)

	# Dictionary containing model accuracy & energy savings for data
	results_dict = {}
	results_dict['LinearRegression'] = {}

	baseline_X = data.loc[:config['separator'], list(set(data.columns).difference(set(['energy'])))]
	projection_X = data.loc[config['separator']:, list(set(data.columns).difference(set(['energy'])))]
	baseline_y = data.loc[:config['separator'], ['energy']]
	projection_y = data.loc[config['separator']:, ['energy']]

	# Run Linear Regression model
	scores = []
	model = LinearRegression()
	kfold = KFold(n_splits=3, shuffle=True, random_state=42)
	for i, (train, test) in enumerate(kfold.split(baseline_X, baseline_y)):
		model.fit(baseline_X.iloc[train], baseline_y.iloc[train])
		scores.append(model.score(baseline_X.iloc[test], baseline_y.iloc[test]))
	mean_score = sum(scores) / len(scores)
	results_dict['LinearRegression']['adj_r2'] = adj_r2(mean_score, baseline_X.shape[0], baseline_X.shape[1])

	# Get predicted values of energy consumption post "separator" time period
	predicted_y = model.predict(projection_X)
	
	# Calculate absolute and percent energy savings
	saving_absolute = (predicted_y - projection_y).sum().values[0]
	saving_perc = (saving_absolute / predicted_y.sum()) * 100
	results_dict['LinearRegression']['Energy Savings (%)'] = float(saving_perc)
	results_dict['LinearRegression']['Energy Savings (absolute)'] = saving_absolute

	return results_dict


def calculate_energy_baselines(df_meter, df_oat, map_uuid_meter, map_uuid_oat):
	""" Main function that preprocesses the data and calculates energy baselines.

	Parameters
	----------
	df_meter 		: pd.DataFrame()
		Meter data.
	df_oat 			: pd.DataFrame()
		Outdoor Air Temperature data.
	map_uuid_meter	: defaultdict(list)
		Mapping of uuid to sitenames (for meter data)
	map_uuid_oat 	: defaultdict(list)
		Mapping of uuid to sitenames (for oat data)

	"""

	# Dictionary containing adjusted r2 values of all sites
	result = {}

	for col_meter in df_meter.columns:

		# Get current sitename
		sitename_meter = map_uuid_meter[col_meter][0]

		# Stores the site data (meter & oat)
		df = pd.DataFrame()
		
		# Join meter data to df and rename column to energy
		df = df.join(df_meter[col_meter], how='outer')
		df.columns = ['energy']

		# Find corresponding data in df_oat and add it to df
		for col_oat in df_oat.columns:
			sitenames_oat = map_uuid_oat[col_oat]
			if sitename_meter in sitenames_oat:
				df = df.join(df_oat[col_oat])

		# Preprocess the site data
		df.index = pd.to_datetime(df.index)
		df = preprocess_data(df)

		result[sitename_meter] = model_data(df)

	return result


if __name__ == '__main__':

	# TO RUN THE APP,
	# Edit parameters in config.json and execute "python app.py".

	# Read json file
	df_meter, df_oat, map_uuid_meter, map_uuid_oat = read_config()

	# Calculate energy baselines
	result_dict = calculate_energy_baselines(df_meter, df_oat, map_uuid_meter, map_uuid_oat)

	print('Result: \n', result_dict)
