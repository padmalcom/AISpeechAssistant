import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import wave
import numpy as np
from scipy.io.wavfile import read

# Example 1 - Temperature time series
series = pd.read_csv('weather_data_4_years.csv', header=0, index_col=0, parse_dates=True)

# convert kelvin to celsius
series['temperature_air_mean_200'] = series['temperature_air_mean_200'].transform(lambda x: x - 273.15)

# Lösche alle Spalten, die wir nicht benötigen
series = series.drop(
	columns=['pressure_air_site', 'temperature_air_mean_005', 'humidity', 'temperature_dew_point_mean_200'],
	errors='ignore'
)

plt.plot(series)
plt.show()

print(series)

result = adfuller(series['temperature_air_mean_200'])
print(result)
is_stationary = False
if result[1] < 0.05:
	is_stationary = True
print('p für Temperaturdaten: ', result[1], " Reihe ist stationär: ", is_stationary)


# Example 2 - Audio wave form
f = read("test.wav")
signal = np.array(f[1], dtype=float)

plt.plot(signal)
plt.show()

result = adfuller(signal)
is_stationary = False
if result[1] < 0.05:
	is_stationary = True
print('p für Waveform: ', result[1], " Reihe ist stationär: ", is_stationary)