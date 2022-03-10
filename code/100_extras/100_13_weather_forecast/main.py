from loguru import logger
from datetime import datetime
from wetterdienst import Settings
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

from wetterdienst.provider.dwd.observation import DwdObservationRequest, DwdObservationPeriod, DwdObservationResolution, DwdObservationParameter, DwdObservationDataset
#Settings.humanize = True

WEATHER_FILE = 'weather_data.csv'
PARAMETERS_FILE = 'parameters.csv'

colors = [
	"blue",
	"orange",
	"green",
	"red",
	"purple",
	"brown",
	"pink",
	"gray",
	"olive",
	"cyan",
]

def get_station():
	stations = DwdObservationRequest(
		parameter=DwdObservationDataset.CLIMATE_SUMMARY,
		resolution=DwdObservationResolution.DAILY,
		period=DwdObservationPeriod.RECENT,
	).filter_by_rank(49.19780976647141, 8.135207205143768, 20).df
	
	print(stations)

	if (len(stations)) > 0:
		return stations.iloc[0]
	return None

def get_data(station):

	request = DwdObservationRequest(
		parameter=DwdObservationDataset.CLIMATE_SUMMARY,
		resolution=DwdObservationResolution.DAILY,
		start_date="2005-01-01",
		end_date="2022-01-01"
	)
	station_data = request.filter_by_station_id(station['station_id']).values.all().df
	parameters =  pd.unique(station_data['parameter']).tolist()
	with open(PARAMETERS_FILE, 'w') as f:
		json.dump(parameters, f, indent=2) 
		
	#station_data = station_data[station_data['parameter'] == 'temperature_air_mean_200']
	
	# convert kelvin to celsius
	#station_data['value'] = station_data['value'].apply(lambda x: x - 273.15)
		
	return station_data, parameters
	
def preprocess_data(station_data, parameters):

	unique_dates =  pd.unique(station_data['date'])
	print("Unique dates: " + str(len(unique_dates)))
	
	
	# drop nan
	station_data.dropna(subset=['value'])
	
	
	aggregegation_functions = {}
	aggregegation_functions['date'] = 'first' # first gets first item in group which is always the same date
	for p in parameters:
		station_data.loc[station_data['parameter'] == p, p] = station_data['value']
		aggregegation_functions[p] = 'sum'
	print(aggregegation_functions)		
	
	station_data = station_data.groupby('date').aggregate(aggregegation_functions)
		
	# delete unused colums
	#station_data.drop('station_id', 1, inplace=True)
	#station_data.drop('dataset', 1, inplace=True)
	#station_data.drop('parameter', 1, inplace=True)

	return station_data
	
def plot_data_step1(station_data, parameters):
	fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k")
	for i in range(len(parameters)):
		key = parameters[i]
		c = colors[i % (len(colors))]
		t_data = station_data[key]
		t_data.index = station_data['date']
		t_data.head()
		ax = t_data.plot(
			ax=axes[i // 2, i % 2],
			color=c,
			title="{} - {}".format(parameters[i], key),
			rot=25,
		)
		ax.legend([parameters[i]])
	plt.tight_layout()
	plt.show()

def main():
	if (not os.path.exists(WEATHER_FILE)):
		station = get_station()
		data, parameters = get_data(station)
		data = preprocess_data(data, parameters)
	else:
		data = pd.read_csv(WEATHER_FILE)
		with open(PARAMETERS_FILE, 'r') as f:
			parameters = json.load(f)
		
	plot_data_step1(data, parameters)
	print(data)
	data.to_csv(WEATHER_FILE, index=False)


if __name__ == "__main__":
	main()