import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import wave
import numpy as np

series = pd.read_csv('weather_data.csv', header=0, index_col=0, parse_dates=True)

# convert kelvin to celsius
series['temperature_air_mean_200'] = series['temperature_air_mean_200'].transform(lambda x: x - 273.15)

# Lösche alle Spalten, die wir nicht benötigen
series = series.drop(columns=['pressure_air_site', 'temperature_air_mean_005', 'humidity', 'temperature_dew_point_mean_200'])

plt.plot(series)
plt.show()

result = adfuller(series)
print('p für Temperaturdaten: ', result[1])

# Lies wav-Datei
f = wave.open("test.wav", "r")
signal = f.readframes(-1)
sample_rate = 16000
signal = np.frombuffer(signal_wave.readframes(f), dtype=np.int16)

plt.plot(series)
plt.show()

result = adfuller(series)
print('p für Waveform: ', result[1])