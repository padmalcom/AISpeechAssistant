# Source = https://keras.io/examples/timeseries/timeseries_weather_forecasting/

from loguru import logger
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tabulate import tabulate

from wetterdienst.provider.dwd.observation import DwdObservationRequest, DwdObservationPeriod, DwdObservationResolution, DwdObservationDataset

WEATHER_FILE = 'weather_data.csv'
PARAMETERS_FILE = 'parameters.csv'

feature_names = ['temperature_air_mean_200']#, 'humidity']

USE_CACHE = False

colors = ["blue", "orange", "green", "red",	"purple", "brown", "pink", "gray", "olive", "cyan"]

# Training parameter
sampling_rate = 6									# sample six data points (6*10 minutes = 1h) to one
sequence_length = 120								# look at 5 days
batch_size = 256
learning_rate = 0.001
epochs=1
outlook=sampling_rate * (sequence_length + 24 - 1)

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
	with open(PARAMETERS_FILE, 'w') as f:
		json.dump(parameters, f, indent=2) 
					
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
	
def plot_raw_data(station_data, parameters):
	fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10), dpi=80, facecolor="w", edgecolor="k")
	for i in range(len(parameters)):
		key = parameters[i]
		c = colors[i % (len(colors))]
		t_data = station_data[key]
		t_data.index = station_data['date']
		t_data.head()
		ax = t_data.plot(
			ax=axes[i // 2, i % 2],
			color=c,
			title="{} - {}".format(parameters[i], key),
			rot=25,
		)
		ax.legend([parameters[i]])
	plt.tight_layout()
	plt.show()
	
def plot_correlation(station_data):
    plt.matshow(station_data.corr())
    plt.xticks(range(station_data.shape[1]), station_data.columns, fontsize=8, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(station_data.shape[1]), station_data.columns, fontsize=8)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=6)
    plt.title("Feature Correlation Heatmap", fontsize=10)
    plt.show()
	
def prepare_training_and_validation_data(station_data, parameters):
	#train_split = int(split_fraction * int(station_data.shape[0]))
	num_training_data = int(0.5 * station_data.shape[0])
	num_validation_data = int(0.25 * station_data.shape[0])
	num_test_data = station_data.shape[0] - num_training_data - num_validation_data
	
	logger.info("Station data length (in days): {}", station_data.shape[0])
	logger.info("Training data: {}, validation data: {}, test data: {}", num_training_data, num_validation_data, num_test_data)

	# Feature value ranges differ, so standardize (form of normalization)
	def standardize(data, num_training_data):
		data_mean = data[:num_training_data].mean(axis=0) # using only training data does not influence val + test
		data_std = data[:num_training_data].std(axis=0)
				
		# return mean and std for inverse standardization
		return (data - data_mean) / data_std, data_mean[0], data_std[0] # Index 0 for temperature
		
	logger.info("Selected parameters are: {}", ", ".join(feature_names))
		
	features = station_data[feature_names]
	features.index = station_data['date']
	logger.info("Indexed data with selected features:\n{}", tabulate(features.head(), headers='keys'))

	features, mean, std = standardize(features.values, num_training_data)
	features = pd.DataFrame(features)
	logger.info("Normalized data:\n{}", tabulate(features.head(), headers='keys'))

	dataset_train = keras.preprocessing.timeseries_dataset_from_array(
		features[:-outlook],#x_train,
		features[outlook:][0],#y_train,
		sequence_length=sequence_length,
		sampling_rate=sampling_rate,
		batch_size=batch_size,
		start_index=0,
		end_index=num_training_data-1
	)

	dataset_val = keras.preprocessing.timeseries_dataset_from_array(
		features[:-outlook],#x_train,
		features[outlook:][0],#y_train,
		sequence_length=sequence_length,
		sampling_rate=sampling_rate,
		batch_size=batch_size,
		start_index=num_training_data,
		end_index=num_training_data + num_validation_data-1
	)
		
	dataset_test = keras.preprocessing.timeseries_dataset_from_array(
		features[:-outlook],
		None,
		sequence_length=sequence_length,
		sampling_rate=sampling_rate,
		batch_size=batch_size,
		start_index=num_training_data + num_validation_data
	)
	
	for batch in dataset_train.take(1):
		inputs, targets = batch

		print("Input shape:", inputs.numpy().shape)
		print("Target shape:", targets.numpy().shape)

	return dataset_train, dataset_val, dataset_test, mean, std

def train(dataset_train, dataset_val):
	for batch in dataset_train.take(1):
		inputs, targets = batch

	logger.info("Input shape: {}", inputs.numpy().shape)
	logger.info("Target shape: {}", targets.numpy().shape)
		
	#inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
	#lstm_out = keras.layers.LSTM(32)(inputs)
	#outputs = keras.layers.Dense(1)(lstm_out)

	#model = keras.Model(inputs=inputs, outputs=outputs)
	#model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
	#model.summary()
	inputs = keras.Input(shape=(sequence_length, inputs.shape[2]))
	x = keras.layers.Bidirectional(keras.layers.LSTM(16))(inputs)
	outputs = keras.layers.Dense(1)(x)
	model = keras.Model(inputs, outputs)

	model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
	history = model.fit(dataset_train,
						epochs=epochs,
						validation_data=dataset_val)
	
	#path_checkpoint = "model_checkpoint.h5"
	#es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

	#modelckpt_callback = keras.callbacks.ModelCheckpoint(
	#	monitor="val_loss",
	#	filepath=path_checkpoint,
	#	verbose=1,
	#	save_weights_only=True,
	#	save_best_only=True,
	#)

	#history = model.fit(
	#	dataset_train,
	#	epochs=epochs,
	#	validation_data=dataset_val,
	#	callbacks=[es_callback, modelckpt_callback],
	#	)
	return model

def plot_data_prediction(plot_data, delta, title):
	labels = ["History", "True Future", "Model Prediction"]
	marker = [".-", "rx", "go"]
	time_steps = list(range(-(plot_data[0].shape[0]), 0))
	
	if delta:
		future = delta
	else:
		future = 0

	plt.title(title)
	
	print(plot_data)
	
	# Plot history (0), true future (1) and model prediction (2)
	for i, val in enumerate(plot_data):
	
		print(i)
		print(plot_data[i])
		# Are we in prediction mode?
		if i == 1 and not val.any():
			continue
			
		if i:
			print("i:", i, " len future: ", future, "len plot data", len(plot_data[i]))
			future = [future] * len(plot_data[i])
			print("i:", i, " len future: ", len(future), "len plot data", len(plot_data[i]))
			plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
		else:
			#plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
			pass
			
	plt.legend()
	print("t1", type(time_steps[0]), "t2:", type((future + 5) * 2))
	plt.xlim([time_steps[0], (future + 5) * 2])
	plt.xlabel("Time-Step")
	plt.show()

	
def inverse_standardization(value, mean, std):
	return value * std + mean
	
def kelvin_to_celsius(value):
	return value - 273.15

def main():
	if (not os.path.exists(WEATHER_FILE) or not USE_CACHE):
		station = get_station()
		data, parameters = get_data(station)
		logger.info("Parameters: {}", parameters)
		data = preprocess_data(data, parameters)
		data.to_csv(WEATHER_FILE, index=False)
	else:
		data = pd.read_csv(WEATHER_FILE)
		with open(PARAMETERS_FILE, 'r') as f:
			parameters = json.load(f)
		
	#plot_raw_data(data, parameters)
	#plot_correlation(data)
		
	dataset_train, dataset_val, dataset_infer, mean, std = prepare_training_and_validation_data(data, parameters)
	
	print("train")
	print(dataset_train)
	
	model = train(dataset_train, dataset_val)
		
	for x, y in dataset_val.take(1):
		print("x:", x)
		print("y:", y)
		#history = x[0][:, 0].numpy() # 0 for temperature
		#ground_truth = y[0].numpy()
		#history = x[0][:, 0].numpy()
		history = x[0][:,0].numpy()
		ground_truth = y.numpy()
		prediction = model.predict(x)[0]

		print(history.shape)
		print(ground_truth.shape)
		print("pred: ", prediction)

				
		history = np.array([kelvin_to_celsius(inverse_standardization(xi, mean, std)) for xi in history])
		ground_truth = np.array([kelvin_to_celsius(inverse_standardization(xi, mean, std)) for xi in ground_truth])
		prediction = [kelvin_to_celsius(inverse_standardization(xi, mean, std)) for xi in prediction]
						
		plot_data_prediction([history, ground_truth, prediction], 12, "Validation")
		
	# predict the actual temperature
	for x in dataset_infer.take(1):
		history = x[0][:, 0].numpy() # 0 for temperature
		#ground_truth = y[0].numpy() # We don't have a ground truth
		prediction = model.predict(x)[0]
				
		history = np.array([kelvin_to_celsius(inverse_standardization(xi, mean, std)) for xi in history])
		prediction = np.array([kelvin_to_celsius(inverse_standardization(xi, mean, std)) for xi in prediction])
						
		plot_data_prediction([history, None, prediction], 12, "Prediction")		
			
if __name__ == "__main__":
	main()