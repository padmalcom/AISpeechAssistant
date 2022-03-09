import logging
from datetime import datetime
from wetterdienst import Settings
import pandas as pd
import matplotlib.pyplot as plt

from wetterdienst.provider.dwd.observation import DwdObservationRequest, DwdObservationPeriod, DwdObservationResolution, DwdObservationParameter, DwdObservationDataset

log = logging.getLogger()

Settings.humanize = True

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
	
	station_data = station_data[station_data['parameter'] == 'temperature_air_mean_200']
	
	# convert kelvin to celsius
	station_data['value'] = station_data['value'].apply(lambda x: x - 273.15)
		
	return station_data

def main():
	logging.basicConfig(level=logging.INFO)
	station = get_station()
	data = get_data(station)
	print(data)
	data.to_csv('weather_data.csv', index=False)
	data.plot(x='date', y='value')
	plt.show()

if __name__ == "__main__":
	main()