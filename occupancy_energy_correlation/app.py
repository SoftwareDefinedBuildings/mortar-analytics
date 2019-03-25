__author__ = "Pranav Gupta"
__email__ = "pranavhgupta@lbl.gov"

import os
import json
import pymortar
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict

""" This app calculates the absolute and percent building energy consumption when the building is occupied. 
Additionally, it creates a plot for each building with its energy consumption data and occupancy data overlapping.

Available aggregations for querying mortar data,
pymortar.MEAN, pymortar.MAX, pymortar.MIN, 
pymortar.COUNT, pymortar.SUM, pymortar.RAW (the temporal window parameter is ignored)

Note
----
1. Set $MORTAR_API_USERNAME and $MORTAR_API_PASSWORD as environment variables.
2. Fetches data from Green Button Meter.

"""

with open('config.json') as f:
	config = json.load(f)


def get_first_last_index(x):
	""" Gets the first and last non-zero index of pandas series row.

	Parameters
	----------
	x 	: pd.Series()
		Row of pandas series.

	Returns
	-------
	pd.Series()
		Contains the start and end index of first non-zero value.

	"""
	start_index = x[x > 0].index[0]
	end_index = x[x > 0].index[-1]
	return pd.Series(dict(start_index=start_index, end_index=end_index))


def occupancy_energy_correlation(df_meter, df_occupancy):
	""" Calculate energy consumption of building when it is occupied.

	Parameters
	----------
	df_meter 		: pd.DataFrame()
		Meter data.
	df_occupancy 	: pd.DataFrame()
		Occupancy data.

	Returns
	-------
	defaultdict
		Result dict containing the energy consumption and % energy consumption when building is occupied.

	"""

	result = defaultdict(lambda: defaultdict(int))

	for site in df_meter.columns:

		occupied_energy = 0

		# Group dataframe by each day and find each day's first and last index where occupancy > 0
		df_temp = pd.DataFrame(df_occupancy[site])
		df_temp2 = df_temp.groupby(df_temp.index.date).apply(get_first_last_index)
		df_temp3 = df_temp2.join(df_temp.groupby(df_temp.index.date).sum())

		# Calculate energy consumption for each day in the time period calculated by first_last_index
		for index, row in df_temp3.iterrows():

			# Calculate energy consumption when building is occupied
			occupied_energy += df_meter.loc[row['start_index']:row['end_index'], [site]].sum()

		total_energy = df_meter[site].sum()

		result[site]['Abs energy consumption'] = occupied_energy
		result[site]['% energy consumption'] = (occupied_energy / total_energy) * 100

		# Plot the two timeseries data
		fig, ax = plt.subplots()
		plot_df = pd.DataFrame()
		plot_df['energy (kWh)'] = df_meter[df_meter.columns[0]]
		plot_df['occupancy'] = df_occupancy[df_occupancy.columns[0]]
		plot_df.dropna(inplace=True)
		plot_df.plot(ax=ax, secondary_y='occupancy', figsize=(18,5))
		plt.savefig(config['results_folder'] + '/' + site + '.png')

	return result


def get_data(response):
	""" Prepares dataframes from the response object of Mortar.

	Parameters
	----------
	response 	: pymortar.result.Result()
		Object returned by client.fetch(request)

	Returns
	-------
	pd.DataFrame(), pd.DataFrame()
		processed meter data, processed occupancy data

	"""

	# Map the uuid's to its sitenames for meter data and occupancy data
	map_uuid_meter, map_uuid_occupancy = defaultdict(list), defaultdict(list)

	resp_meter = response.query('select * from view_meter')
	resp_occupancy = response.query('select * from view_occupancy')

	for (url, uuid, sitename) in resp_meter:
		map_uuid_meter[uuid].append(sitename)
	for (url, uuid, sitename) in resp_occupancy:
		map_uuid_occupancy[uuid].append(sitename)

	# Get dataframes
	df_meter = response['data_meter']
	df_occupancy = response['data_occupancy']

	# Drop null values
	df_meter.dropna(inplace=True)
	df_occupancy.dropna(inplace=True)

	print('df_meter: \n', df_meter)
	print('df_occupancy: \n', df_occupancy)

	# Remove outliers that are 3 std-dev away from mean
	df_meter = df_meter[(np.abs(stats.zscore(df_meter)) < float(3)).all(axis=1)]
	df_occupancy = df_occupancy[(np.abs(stats.zscore(df_occupancy)) < float(3)).all(axis=1)]

	# Change uuid to sitename for meter data columns
	df_meter.columns = [''.join(map_uuid_meter[col]) for col in df_meter.columns]


	# A building will have multiple occupancy sensors in it
	# Map building name to the list of its occupancy sensors
	sitename_uuids = defaultdict(list)

	# dict[sitename] = [list of uuids]
	for sitename in df_meter.columns:
		for uuid_occupancy in df_occupancy.columns:
			if ''.join(map_uuid_occupancy[uuid_occupancy]) == sitename:
				sitename_uuids[sitename].append(uuid_occupancy)

	# Create new dataframe that contains the sum of all occupancy sensors in a site/building
	new_df_occupancy = pd.DataFrame()
	for sitename, list_uuids in sitename_uuids.items():
		# Sum across all uuids belonging to a site (axis=1 sums across columns)
		new_df_occupancy[sitename] = df_occupancy[list_uuids].sum(axis=1)

	return df_meter, new_df_occupancy


def read_config():
	""" Reads config.json file to obtain parameters and fetch data from Mortar.

	Returns
	-------
	pd.DataFrame(), pd.DataFrame(), default(list), default(list)
		meter data, occupancy data, map of uuid to meter data, map of uuid to occupancy data
	
	"""

	# Instatiate Client
	client = pymortar.Client({})

	# Query for meter data
	query_meter = "SELECT ?meter WHERE { ?meter rdf:type/rdfs:subClassOf* brick:Green_Button_Meter };"

	# Query for occupancy data
	query_occupancy = "SELECT ?point WHERE { ?point rdf:type/rdfs:subClassOf* brick:Occupancy_Sensor };"

	# Get list of sites for meter data and occupancy data
	resp_meter = client.qualify([query_meter])
	resp_occupancy = client.qualify([query_occupancy])

	if resp_meter.error or resp_occupancy.error:
		print("ERORR: ", resp_meter.error if True else resp_occupancy.error)
		os._exit(0)
	else:

		# Get list of sites that are common for meter data and occupancy data
		common_sites = list(set(resp_meter.sites).intersection(set(resp_occupancy.sites)))

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
	occupancy = pymortar.View(
		name="view_occupancy",
		sites=config['sites'],
		definition=query_occupancy
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

	# Define the occupancy timeseries stream
	data_view_occupancy = pymortar.DataFrame(
		name="data_occupancy", # dataframe column name
		aggregation=pymortar.RAW,
		window="",
		timeseries=[
			pymortar.Timeseries(
				view="view_occupancy",
				dataVars=["?point"]
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
		views=[meter, occupancy],
		dataFrames=[data_view_meter, data_view_occupancy],
		time=time_params
	)

	# Fetch data from request
	response = client.fetch(request)

	# Save data to csv file
	if config['save_data']:
		response['data_meter'].to_csv('meter_data.csv')
		response['data_occupancy'].to_csv('occupancy_data.csv')

	# Create results folder if it doesn't exist
	if not os.path.exists('./' + config['results_folder']):
		os.mkdir('./' + config['results_folder'])

	return response


if __name__ == '__main__':

	# TO RUN THE APP,
	# Edit parameters in config.json and execute "python app.py".

	# Read json file
	response = read_config()

	# Get data from the response object
	df_meter, df_occupancy = get_data(response)

	# Calculate correlation between building occupancy and energy consumption
	result = occupancy_energy_correlation(df_meter2, df_occupancy2)

	# Absolute energy consumption is the energy consumption when the building is occupied
	# % energy consumption tells the percent of building energy consumed when it is occupied
	print('Result: \n', dict(result))
