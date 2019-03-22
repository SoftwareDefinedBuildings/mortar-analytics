__author__ = "Pranav Gupta"
__email__ = "pranavhgupta@lbl.gov"

import os
import json
import pymortar
import pandas as pd
from collections import defaultdict

""" This app calculates the building energy when it is unoccupied.

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
	resp_occupancy = data.query('select * from view_occupancy')

	map_uuid_meter, map_uuid_occupancy = defaultdict(list), defaultdict(list)

	for (url, uuid, sitename) in resp_meter:
		map_uuid_meter[uuid].append(sitename)
	for (url, uuid, sitename) in resp_occupancy:
		map_uuid_occupancy[uuid].append(sitename)

	return map_uuid_meter, map_uuid_occupancy


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
		name="data_occupancy",
		aggregation=pymortar.MEAN,
		window="15m",
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
	data = client.fetch(request)

	# Renames columns from uuids' to sitenames'
	map_uuid_meter, map_uuid_occupancy = map_uuid_sitename(data)

	# Save data to csv file
	if config['save_data']:
		data['data_meter'].to_csv('meter_data.csv')
		data['data_occupancy'].to_csv('occupancy_data.csv')

	return data['data_meter'], data['data_occupancy'], map_uuid_meter, map_uuid_occupancy


def preprocess_data(df_meter, df_occupancy, map_uuid_meter, map_uuid_occupancy):
	""" Process the data - dropna's and add up occupancy data from all endpoints for a single building.

	Parameters
	----------
	df_meter 		: pd.DataFrame()
		Meter data.
	df_occupancy 	: pd.DataFrame()
		Occupancy data.

	Returns
	-------
	pd.DataFrame(), pd.DataFrame()
		processed meter data, processed occupancy data

	"""

	# Drop null values
	df_meter.dropna(inplace=True)
	df_occupancy.dropna(inplace=True)

	# Change uuid to sitename for meter data
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
		new_df_occupancy[sitename] = df_occupancy[list_uuids].sum(axis=1)

	return df_meter, new_df_occupancy


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
		Result dict containing the total energy consumption and unoccupied energy consumption.

	"""

	# Dictionary containing adjusted r2 values of all sites
	result = {}

	for sitename in df_meter.columns:

		# Create dataframe of a site's energy & occupancy
		df = pd.DataFrame(df_meter[sitename].copy())
		df = df.join(df_occupancy[sitename])
		df.columns = ['energy', 'occupancy']

		# Calculate total energy and occupancy
		total_energy = df['energy'].sum()
		total_occupancy = df['occupancy'].sum()

		# Calculate absolute & percent energy when building is unoccupied
		unoccupied_energy_abs = df[df['occupancy'] == 0].sum()
		unoccupied_energy_perc = (unoccupied_energy_abs.values[0] / total_energy) * 100

		result[sitename] = {}
		result[sitename]['Total Energy Consumption'] = total_energy
		result[sitename]['Total Occupancy'] = total_occupancy
		result[sitename]['Unoccupied Energy Consumption (absolute)'] = unoccupied_energy_abs.values[0]
		result[sitename]['Unoccupied Energy Consumption (percent)'] = unoccupied_energy_perc

	return result


if __name__ == '__main__':

	# TO RUN THE APP,
	# Edit parameters in config.json and execute "python app.py".

	# Read json file
	df_meter, df_occupancy, map_uuid_meter, map_uuid_occupancy = read_config()

	# Process the data before doing the main analysis
	df_meter, df_occupancy = preprocess_data(df_meter, df_occupancy, map_uuid_meter, map_uuid_occupancy)

	# Calculate correlation between building occupancy and energy consumption
	result = occupancy_energy_correlation(df_meter, df_occupancy)

	print('Result: \n', result)
