import os
import csv
import pandas as pd
from matplotlib import pyplot as plt
from loguru import logger
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
from sklearn.metrics import mean_squared_error

# Rerefenz: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

WEATHER_DATA = os.path.join("weather_data_1_year.csv")

if __name__ == '__main__':

	# Lese Daten und interpretiere Spalte 'date' als Index
	data = pd.read_csv(WEATHER_DATA, index_col='date', usecols=['date', 'temperature_air_mean_200'])

	# Rechne Kelvin in Grad Celsius um
	data['temperature_air_mean_200'] = data['temperature_air_mean_200'] - 273.15
	
	# Behalte nur die Tagestemperatur Mittags, 12 Uhr
	data.index = pd.to_datetime(data.index)
	m = (data.index.minute == 0) & (data.index.second == 0) & (data.index.hour == 12)
	data = data[m]
	
	# Passe das Datumsformat an
	data.index = pd.to_datetime(data.index.strftime('%Y-%m-%d'))
	
	logger.info("Anzahl Datensätze {}.", len(data))
	print(data.head())
	
	X = data.values
	size = int(len(X) * 0.75)
	train, test = X[0:size], X[size:len(X)]
	history = [x for x in train]
	predictions = list()
	
	# Forwärtsgerichtete Vorhersage jeweils des nächsten Wertes
	for t in range(len(test)):
	
		# Die Ordner des Arima-Modells werden unoptimiert auf
		# 1,1,1 festgelegt (Siehe Hinweiskasten)
		model = ARIMA(history, order=(1,1,1))
		model_fit = model.fit()
		output = model_fit.forecast()
		yhat = output[0]
		predictions.append(yhat)
		obs = test[t]
		
		# Füge die vorhergesagte Temperatur der Datengrundlage für
		# das nächste Training hinzu.
		history.append(obs)
		
		logger.info("Vorhersage: {}, tatsächlicher Wert: {}", yhat, obs)
		
	# Berechne Fehler der Vorhersage
	rmse = sqrt(mean_squared_error(test, predictions))
	logger.info("Root-Mean-Squared Error: {}", rmse)
	
	# Visualisierung der Güte des Modells
	plt.plot(test)
	plt.plot(predictions, color='red')
	plt.show()