# Source = https://keras.io/examples/timeseries/timeseries_weather_forecasting/

from loguru import logger
from datetime import datetime
#from wetterdienst import Settings
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import tensorflow as tf
from tensorflow import keras
import numpy as np

from wetterdienst.provider.dwd.observation import DwdObservationRequest, DwdObservationPeriod, DwdObservationResolution, DwdObservationDataset
#Settings.humanize = True

WEATHER_FILE = 'weather_data.csv'
PARAMETERS_FILE = 'parameters.csv'

colors = [
	"blue",
	"orange",
	"green",
	"red",
	"purple",
	"brown",
	"pink",
	"gray",
	"olive",
	"cyan",
]

# Training parameter
split_fraction = 0.75 # 75 percent of the data is used for training
step = 1 # this samples 6 points (1 = 10 minutes) to 1 (1 hour), since we have hours, we don't need this
past = 120 #Look in the past 120 hours to predict... # 720 hours
future = 1 #the next 12 hours in the future # 72 hours
learning_rate = 0.001
batch_size = 256
epochs = 10

def get_station():
	stations = DwdObservationRequest(
		parameter=DwdObservationDataset.CLIMATE_SUMMARY,
		resolution=DwdObservationResolution.DAILY,
		period=DwdObservationPeriod.RECENT,
	).filter_by_rank(49.19780976647141, 8.135207205143768, 20).df
	
	print(stations)

	if (len(stations)) > 0:
		return stations.iloc[0]
	return None

def get_data(station):

	request = DwdObservationRequest(
		parameter=DwdObservationDataset.CLIMATE_SUMMARY,
		resolution=DwdObservationResolution.DAILY,
		start_date="2005-01-01",
		end_date="2022-01-01"
	)
	station_data = request.filter_by_station_id(station['station_id']).values.all().df
	parameters =  pd.unique(station_data['parameter']).tolist()
	with open(PARAMETERS_FILE, 'w') as f:
		json.dump(parameters, f, indent=2) 
					
	return station_data, parameters
	
def preprocess_data(station_data, parameters):

	unique_dates =  pd.unique(station_data['date'])
	print("Unique dates: " + str(len(unique_dates)))
		
	# drop nan
	station_data.dropna(subset=['value'])
		
	aggregegation_functions = {}
	aggregegation_functions['date'] = 'first' # first gets first item in group which is always the same date
	for p in parameters:
		station_data.loc[station_data['parameter'] == p, p] = station_data['value']
		aggregegation_functions[p] = 'sum'
	print(aggregegation_functions)		
	
	station_data = station_data.groupby('date').aggregate(aggregegation_functions)
		
	return station_data
	
def plot_raw_data(station_data, parameters):
	fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k")
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
    plt.xticks(range(station_data.shape[1]), station_data.columns, fontsize=10, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(station_data.shape[1]), station_data.columns, fontsize=10)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    plt.title("Feature Correlation Heatmap", fontsize=10)
    plt.show()

def prepare_training_and_validation_data(station_data, parameters):
	train_split = int(split_fraction * int(station_data.shape[0]))

	# normalize data to range [0, 1]
	def normalize(data, train_split):
		data_mean = data[:train_split].mean(axis=0)
		data_std = data[:train_split].std(axis=0)
		print(data_mean)
		print(data_std)
		return (data - data_mean) / data_std, data_mean[2], data_std[2] # Index 2 for temperature
		
	print("The selected parameters are:", ", ".join([parameters[i] for i in [5-1, 8-1, 10-1, 11-1]])) # 10-1 because date is still in the dataframe
	selected_features = [parameters[i] for i in [5-1, 8-1, 10-1, 11-1]]
	features = station_data[selected_features]
	features.index = station_data['date']
	print(features.head())

	features, mean, std = normalize(features.values, train_split)
	features = pd.DataFrame(features)
	print(features.head())

	train_data = features.loc[0 : train_split - 1]
	print("Train data count: " + str(len(train_data)))
	val_data = features.loc[train_split:]

	# training dataset
	start = past + future
	end = start + train_split

	x_train = train_data[[i for i in range(len(selected_features))]].values
	y_train = features.iloc[start:end][[2]] # Training in mean temperature (index 2)

	sequence_length = int(past / step)

	dataset_train = keras.preprocessing.timeseries_dataset_from_array(
		x_train,
		y_train,
		sequence_length=sequence_length,
		sampling_rate=step,
		batch_size=batch_size,
	)

	# validation dataset
	x_end = len(val_data) - past - future

	label_start = train_split + past + future

	x_val = val_data.iloc[:x_end][[i for i in range(len(selected_features))]].values
	y_val = features.iloc[label_start:][[2]]

	dataset_val = keras.preprocessing.timeseries_dataset_from_array(
		x_val,
		y_val,
		sequence_length=sequence_length,
		sampling_rate=step,
		batch_size=batch_size,
	)
	
	return dataset_train, dataset_val, mean, std

def train(dataset_train, dataset_val):
	for batch in dataset_train.take(1):
		inputs, targets = batch

	print("Input shape:", inputs.numpy().shape)
	print("Target shape:", targets.numpy().shape)
		
	inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
	lstm_out = keras.layers.LSTM(32)(inputs)
	outputs = keras.layers.Dense(1)(lstm_out)

	model = keras.Model(inputs=inputs, outputs=outputs)
	model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
	model.summary()
	
	path_checkpoint = "model_checkpoint.h5"
	es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

	modelckpt_callback = keras.callbacks.ModelCheckpoint(
		monitor="val_loss",
		filepath=path_checkpoint,
		verbose=1,
		save_weights_only=True,
		save_best_only=True,
	)

	history = model.fit(
		dataset_train,
		epochs=epochs,
		validation_data=dataset_val,
		callbacks=[es_callback, modelckpt_callback],
	)
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
	for i, val in enumerate(plot_data):
		if i:
			plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
		else:
			plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
	plt.legend()
	plt.xlim([time_steps[0], (future + 5) * 2])
	plt.xlabel("Time-Step")
	plt.show()
	return
	
def human_readable(value, mean, std):
	new_value = (value * std + mean) # - 273.15
	
	

def main():
	if (not os.path.exists(WEATHER_FILE)):
		station = get_station()
		data, parameters = get_data(station)
		data = preprocess_data(data, parameters)
		data.to_csv(WEATHER_FILE, index=False)
	else:
		data = pd.read_csv(WEATHER_FILE)
		with open(PARAMETERS_FILE, 'r') as f:
			parameters = json.load(f)
		
	#plot_raw_data(data, parameters)
	#plot_correlation(data)
	
	dataset_train, dataset_val, mean, std = prepare_training_and_validation_data(data, parameters)
	
	model = train(dataset_train, dataset_val)
	
	for x, y in dataset_val.take(5):
		history = x[0][:, 1].numpy()
		ground_truth = y[0].numpy()
		prediction = model.predict(x)[0]
		
		history = np.array([human_readable(xi, mean, std) for xi in history])
		ground_truth = np.array([human_readable(xi, mean, std) for xi in ground_truth])
		prediction = np.array([human_readable(xi, mean, std) for xi in prediction])
				
		plot_data_prediction([history, ground_truth, prediction], 12, "Single Step Prediction")
	
if __name__ == "__main__":
	main()