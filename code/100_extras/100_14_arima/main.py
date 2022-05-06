import os
import csv
from wetterdienst.provider.dwd.observation import DwdObservationRequest, DwdObservationPeriod, DwdObservationResolution, DwdObservationDataset
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate
from loguru import logger

import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# only display tensorflow error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

WEATHER_DATA = os.path.join("weather_data_1_year.csv")

def read_data():
	temperature = []
	raw_data = []
	dates = []
	with open(WEATHER_DATA, newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='\'')
			
		for i, row in enumerate(spamreader):
			if i==0:
				continue
			row[2] = float(row[2]) - 273.15 # kelvin to celsius
			temperature.append(float(row[2]))
			raw_data.append([float(x) for x in row[2:6]])
			dates.append(row[0])
			
	logger.info("Raw data (first 5 entries):\n{}", raw_data[:5])
	logger.info("Temperature (first 5 entries):\n{}", temperature[:5])

	num_train_samples = int(0.7 * len(raw_data))
	num_val_samples = int(0.2 * len(raw_data))
	
	logger.info("Number of training samples: {}, number of evaluation samples: {}, remaining samples for prediction: {}",
		num_train_samples,
		num_val_samples,
		num_train_samples - num_val_samples
	)
	return raw_data, temperature, dates, num_train_samples, num_val_samples



if __name__ == '__main__':
	data = pd.read_csv(WEATHER_DATA, index_col='date', usecols=['date', 'temperature_air_mean_200'])
	data['temperature_air_mean_200'] = data['temperature_air_mean_200'] - 273.15

	data = data[:300]
	train, test = train_test_split(data, train_size=200, test_size=100)
	
	# Fit your model
	model = pm.auto_arima(train['temperature_air_mean_200'], seasonal=True, m=12) # 10 Minuten Serien
	# make your forecasts
	forecasts = model.predict(test.shape[0])  # predict N steps into the future

	# Visualize the forecasts (blue=train, green=forecasts)
	x = np.arange(data.shape[0])
	plt.plot(x[:200], train['temperature_air_mean_200'], c='blue')
	plt.plot(x[200:], forecasts, c='green')
	plt.plot(x[200:], test['temperature_air_mean_200'], c='orange')
	plt.show()