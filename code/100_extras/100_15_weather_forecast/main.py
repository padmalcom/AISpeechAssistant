import os
import csv
from wetterdienst.provider.dwd.observation import DwdObservationRequest, DwdObservationPeriod, DwdObservationResolution, DwdObservationDataset
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tabulate import tabulate
from loguru import logger

# only display tensorflow error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

WEATHER_DATA = os.path.join("weather_data.csv")

SAMPLING_RATE = 6
SEQUENCE_LENGTH = 120
OUTLOOK = SAMPLING_RATE * (SEQUENCE_LENGTH + 24*3 - 1) # 3 days in the future
BATCH_SIZE = 256
EPOCHS = 15

def get_station():

	start_date = (datetime.now() - timedelta(days=1*365)).strftime('%Y-%m-%d')
	end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
	
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
	
def get_data(station):

	start_date = (datetime.now() - timedelta(days=1*365)).strftime('%Y-%m-%d')
	end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
	
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

def standardize(raw_data, temperature):
	raw_data = np.array(raw_data[:])
	mean = raw_data[:num_train_samples].mean(axis=0)
	std = raw_data[:num_train_samples].std(axis=0)
	logger.info("Standardizing data with mean {} and standard deviation {}.", mean, std)
	raw_data = (raw_data - mean) / std
	temperature = [(x - mean[0]) / std[0] for x in temperature]
	return raw_data, temperature, mean, std

def create_datasets(raw_data, temperature, num_train_samples, num_val_samples):
	train_dataset = keras.utils.timeseries_dataset_from_array(
		raw_data[:-OUTLOOK],
		targets=temperature[OUTLOOK:],
		sampling_rate=SAMPLING_RATE,
		sequence_length=SEQUENCE_LENGTH,
		shuffle=True,
		batch_size=BATCH_SIZE,
		start_index=0,
		end_index=num_train_samples)

	val_dataset = keras.utils.timeseries_dataset_from_array(
		raw_data[:-OUTLOOK],
		targets=temperature[OUTLOOK:],
		sampling_rate=SAMPLING_RATE,
		sequence_length=SEQUENCE_LENGTH,
		shuffle=True,
		batch_size=BATCH_SIZE,
		start_index=num_train_samples,
		end_index=num_train_samples + num_val_samples - 1)
		
	for samples, targets in train_dataset.take(1):
		logger.info("Samples shape: {} (num samples, sequence length, num features)", samples.shape)
		logger.info("targets shape: {} (normalized temperature)", targets.shape)
		
	return train_dataset, val_dataset

def train(feature_count, train_dataset, val_dataset):
	inputs = keras.Input(shape=(SEQUENCE_LENGTH, raw_data.shape[-1]))
	x = layers.Bidirectional(layers.LSTM(32))(inputs)
	outputs = layers.Dense(1)(x)
	model = keras.Model(inputs, outputs)

	model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
	history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)
						
	loss = history.history["mae"]
	val_loss = history.history["val_mae"]
	epochs = range(1, len(loss) + 1)
	return model, loss, val_loss, epochs


def visualize_training(loss, val_loss, epochs):
	plt.figure()
	plt.plot(epochs, loss, "bo", label="Training MAE")
	plt.plot(epochs, val_loss, "b", label="Validation MAE")
	plt.title("Training and validation MAE")
	plt.legend()
	plt.show()

def visualize_evaluation(plot_data, delta, title):
	labels = ["History", "True Future", "Model Prediction"]
	marker = [".-", "rx", "go"]
	time_steps = list(range(-(plot_data[0].shape[0]), 0))
	if delta:
		future = delta
	else:
		future = 0

	plt.title(title)
	for i, val in enumerate(plot_data):
		if i: # i==1: ground_truth, i==2: prediction
			plt.plot(future, plot_data[i][0,0], marker[i], markersize=10, label=labels[i])
		else: # i== 0: history
			plt.plot(time_steps, [x[0] for x in plot_data[i]], marker[i], label=labels[i])
	plt.legend()
	plt.xlim([time_steps[0], (future + 5) * 2])
	plt.xlabel("Time-Step")
	plt.show()

def evaluate(val_dataset, model):
	for x, y in val_dataset.take(1):
		history = x[0][:, 0].numpy()
		ground_truth = y[0].numpy()
		prediction = model.predict(x)[0]

		history = np.array([xi * std  + mean for xi in history])
		ground_truth = np.array([ground_truth * std + mean])
		prediction = np.array([prediction * std + mean])
				
		visualize_evaluation([history, ground_truth, prediction], 3, "Single Step Prediction") # 3 = 3 day in the future

def predict(prediction_sequence):
	prediction = model.predict(prediction_sequence)
	prediction = np.array([prediction * std + mean])
	prediction = prediction[0]
	return prediction[0,0]

if __name__ == '__main__':

	if not os.path.exists(WEATHER_DATA):
		station = get_station()
		logger.info("Found station {}.", station)
		data, parameters = get_data(station)
		logger.info("Read data with {} entries.", len(data))
		logger.info("Parameters: {}", parameters)
		data = preprocess_data(data, parameters)
		data.to_csv(WEATHER_DATA, index=False)
		logger.info("Weather file {} written.", WEATHER_DATA)
		
	logger.info("GPUs Available: {}", len(tf.config.list_physical_devices('GPU')))
	
	raw_data, temperature, dates, num_train_samples, num_val_samples = read_data()
	raw_data, temperature, mean, std = standardize(
		raw_data,
		temperature
	)
	train_dataset, val_dataset = create_datasets(
		raw_data,
		temperature,
		num_train_samples,
		num_val_samples
	)
	model, loss, val_loss, epochs = train(
		raw_data.shape[-1],
		train_dataset,
		val_dataset
	)
	visualize_training(
		loss,
		val_loss,
		epochs
	)
	evaluate(
		val_dataset,
		model
	)
	
	# hour that we want to predict the time for on the next day
	prediction_hour = 12

	# Create sequence for prediction from last 5 days (120 hours) to predict the next 72 hours
	prediction_sequence = np.asarray(raw_data[-120-prediction_hour*6:-prediction_hour*6]) # need exactly sequence of 120 entries
	
	# Insert one dimension (# of samples)
	prediction_sequence = np.expand_dims(prediction_sequence, axis=0) 
	weather_in_24h_day = predict(prediction_sequence)
	
	# get the last date in the dataset and add 3 days
	tomorrow = datetime.fromisoformat(dates[-1  - prediction_hour*6]) + timedelta(days=3)

	logger.info("Temperature at {} is {}.", tomorrow, weather_in_24h_day)
	
	
	
	
	
	
	
	