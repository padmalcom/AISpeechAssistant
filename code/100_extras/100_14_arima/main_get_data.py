import os
import csv
from wetterdienst.provider.dwd.observation import DwdObservationRequest, DwdObservationPeriod, DwdObservationResolution, DwdObservationDataset
from datetime import datetime
import pandas as pd
from tabulate import tabulate
from loguru import logger

WEATHER_DATA = os.path.join("weather_data.csv")

def get_station(start_date, end_date):
	
	stations = DwdObservationRequest(
		parameter=[DwdObservationDataset.TEMPERATURE_AIR],
		resolution=DwdObservationResolution.MINUTE_10,
		start_date=start_date,
		end_date=end_date
	
	).filter_by_rank(49.19780976647141, 8.135207205143768, 20).df
	
	logger.info("\n"+tabulate(stations, headers='keys'))

	if (len(stations)) > 0:
		return stations.iloc[0]
	return None
	
def get_data(station, start_date, end_date):
	
	logger.info("start date: {} ", start_date)
	logger.info("end date: {}", end_date)
	
	request = DwdObservationRequest(
		parameter=[DwdObservationDataset.TEMPERATURE_AIR],
		resolution=DwdObservationResolution.MINUTE_10,
		start_date=start_date,
		end_date=end_date
	)
	station_data = request.filter_by_station_id(station['station_id']).values.all().df
	parameters =  pd.unique(station_data['parameter']).tolist()					
	return station_data, parameters
	
def preprocess_data(station_data, parameters):

	unique_dates =  pd.unique(station_data['date'])
	logger.info("Unique dates: {} ", len(unique_dates))
		
	# drop nan
	station_data.dropna(subset=['value'])
		
	aggregegation_functions = {}
	aggregegation_functions['date'] = 'first' # first gets first item in group which is always the same date
	for p in parameters:
		station_data.loc[station_data['parameter'] == p, p] = station_data['value']
		aggregegation_functions[p] = 'sum'
	logger.info("Aggregation function: {}", aggregegation_functions)		
	
	station_data = station_data.groupby('date').aggregate(aggregegation_functions)
		
	return station_data

if __name__ == '__main__':

	start_date = datetime.fromisoformat('2021-01-01 00:00:00')
	end_date = datetime.fromisoformat('2021-06-01 23:59:59')

	station = get_station(start_date, end_date)
	logger.info("Found station {}.", station)
	data, parameters = get_data(station, start_date, end_date)
	logger.info("Read data with {} entries.", len(data))
	logger.info("Parameters: {}", parameters)
	data = preprocess_data(data, parameters)
	data.to_csv(WEATHER_DATA, index=False)
	logger.info("Weather file {} written.", WEATHER_DATA)
	
	
	
	
	