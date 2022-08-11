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

# Zeige nur TensorFlow-Error Nachrichten
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

WEATHER_DATA = os.path.join("weather_data.csv")

SAMPLING_RATE = 6
SEQUENCE_LENGTH = 120
OUTLOOK = SAMPLING_RATE * (SEQUENCE_LENGTH + 24*3 - 1) # 3 Tage in der Zukunft
BATCH_SIZE = 256
EPOCHS = 15

def read_data():
	temperature = []
	raw_data = []
	dates = []
	with open(WEATHER_DATA, newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='\'')
			
		for i, row in enumerate(spamreader):
			if i==0:
				continue
			row[2] = float(row[2]) - 273.15 # Kelvin zu Celsius
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
	plt.title("Mittlere absolute Fehler von Training und Validation")
	plt.legend()
	plt.show()

def visualize_evaluation(plot_data, delta, title):
	labels = ["History", "Ground Truth", "Model Prediction"]
	marker = [".-", "rx", "go"]
	time_steps = list(range(-(plot_data[0].shape[0]), 0))
	if delta:
		future = delta
	else:
		future = 0

	plt.title(title)
	for i, val in enumerate(plot_data):
		if i: # i==1: Ground Truth, i==2: Prediction
			plt.plot(future, plot_data[i][0,0], marker[i], markersize=10, label=labels[i])
		else: # i== 0: History
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
				
		 # Markiere 3 Tage in der Zukunft
		visualize_evaluation([history, ground_truth, prediction], 3*24, "3-Tages-Temperaturprognose")

def predict(prediction_sequence):
	prediction = model.predict(prediction_sequence)
	prediction = np.array([prediction * std + mean])
	prediction = prediction[0]
	return prediction[0,0]

if __name__ == '__main__':
		
	logger.info("GPUs verfügbar: {}", len(tf.config.list_physical_devices('GPU')))
	
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
	
	# Genaue Stunde, an der die Temperatur in 3 Tagen vorhergesagt werden soll
	prediction_hour = 12

	# Generieren Sequenz der letzten 5 Tage (120 Stunden), um die nächsten 72 Stunden vorherzusagen
	prediction_sequence = np.asarray(raw_data[-120-prediction_hour*6:-prediction_hour*6])
	
	# Füge eine Dimension hinzu (Anzahl Samples)
	# Insert one dimension (# of samples)
	print("Before: ", prediction_sequence)
	prediction_sequence = np.expand_dims(prediction_sequence, axis=0) 
	print("After: ", prediction_sequence)
	weather_in_24h_day = predict(prediction_sequence)
	
	# Hole das letzte Datum des Datensatzes und füge 3 Tage hinzu.
	tomorrow = datetime.fromisoformat(dates[-1  - prediction_hour*6]) + timedelta(days=3)

	logger.info("Temperatur am {} ist {}.", tomorrow, weather_in_24h_day)
	
	
	
	
	
	
	
	