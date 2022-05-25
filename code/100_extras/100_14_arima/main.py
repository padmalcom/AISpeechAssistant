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

	
	train, test = train_test_split(data, train_size=.90, test_size=.1)
	print("Len train data", len(train), "len test data", len(test))

	print(train)
	# Fit your model
	model = pm.auto_arima(train, start_p=0, d=0, start_q=0, max_p=5, max_d=5, max_q=5, start_P=0, D=1, start_Q=0, max_P=5, max_D=5,max_Q=5, n_fits=50, seasonal=False) # Tagesserien
	print(model.summary())
	# make your forecasts
	forecasts = model.predict(test.shape[0])  # predict N steps into the future
	
	print(test.head())
	print("len test", len(test))
	print("fc", forecasts)

	# Visualize the forecasts (blue=train, green=forecasts)
	#x = np.arange(data.shape[0])
	#plt.plot(x[:len(train)], train['temperature_air_mean_200'], c='blue')
	#plt.plot(x[len(train):], forecasts, c='green')
	#plt.plot(x[len(train):], test['temperature_air_mean_200'], c='orange')
	
	plt.plot(data.index[:len(train)], train['temperature_air_mean_200'], c='blue')
	plt.plot(data.index[len(train):], forecasts, c='green')
	plt.plot(data.index[len(train):], test['temperature_air_mean_200'], c='orange')
	plt.show()
	
	
	from statsmodels.tsa.arima.model import ARIMA
	predictions = list()
	# walk-forward validation
	history = train.copy()
	print(history)
	for t in range(len(test)):
		model = ARIMA(history, order=(1,1,1))
		model_fit = model.fit()
		output = model_fit.forecast()
		yhat = output[0]
		predictions.append(yhat)
		obs = test[t]
		history.append(obs)
		print('predicted=%f, expected=%f' % (yhat, obs))
	pyplot.plot(test)
	pyplot.plot(predictions, color='red')
	pyplot.show()