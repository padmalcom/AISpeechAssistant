import os
import csv
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
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
from sklearn.metrics import mean_squared_error

# https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

WEATHER_DATA = os.path.join("weather_data_2_years.csv")

if __name__ == '__main__':

	# read data and parse columns
	data = pd.read_csv(WEATHER_DATA, index_col='date', usecols=['date', 'temperature_air_mean_200'])

	# convert fahrenheit to celsius
	data['temperature_air_mean_200'] = data['temperature_air_mean_200'] - 273.15
	
	# keep only every measurement at 12 am
	data.index = pd.to_datetime(data.index)
	m = (data.index.minute == 0) & (data.index.second == 0) & (data.index.hour == 12)
	data = data[m]
	data.index = pd.to_datetime(data.index.strftime('%Y-%m-%d'))
	
	print(len(data))
	print(data.head())

	
	X = data.values
	size = int(len(X) * 0.66)
	train, test = X[0:size], X[size:len(X)]
	history = [x for x in train]
	predictions = list()
	# walk-forward validation
	for t in range(len(test)):
		model = ARIMA(history, order=(1,1,1))
		model_fit = model.fit()
		output = model_fit.forecast()
		yhat = output[0]
		predictions.append(yhat)
		obs = test[t]
		history.append(obs)
		print('predicted=%f, expected=%f' % (yhat, obs))
	# evaluate forecasts
	rmse = sqrt(mean_squared_error(test, predictions))
	print('Test RMSE: %.3f' % rmse)
	# plot forecasts against actual outcomes
	plt.plot(test)
	plt.plot(predictions, color='red')
	plt.show()