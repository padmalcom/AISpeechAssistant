import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# load dataset
#def parser(x):
#	return datetime.strptime('190'+x, '%Y-%m')

series = pd.read_csv('weather_data.csv', header=0, index_col=0, parse_dates=True)

# convert kelvin to celsius
series['temperature_air_mean_200'] = series['temperature_air_mean_200'].transform(lambda x: x - 273.15)

series.index = series.index.to_period('10m')

print(series.head())

# fit model
model = ARIMA(series['temperature_air_mean_200'], order=(5,1,0))

import sys
sys.exit()

model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
# density plot of residuals
residuals.plot(kind='kde')
pyplot.show()
# summary stats of residuals
print(residuals.describe())