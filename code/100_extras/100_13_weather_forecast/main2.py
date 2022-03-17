import os
import csv
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

fname = os.path.join("weather_data.csv")

temperature = []
dates = []
raw_data = []

with open(fname, newline='') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='\'')
		
	for i, row in enumerate(spamreader):
		if i==0:
			continue
		
		dates.append(datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S%z')) #2021-03-16 00:00:00+00:00
		temperature.append(float(row[2]))
		raw_data.append([float(x) for x in row[2:6]])

print(len(dates), len(temperature))
#plt.plot(dates, temperature)
#plt.show()

#plt.plot(dates[:1440], temperature[:1440])
#plt.show()

num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples
print("num_train_samples:", num_train_samples)
print("num_val_samples:", num_val_samples)
print("num_test_samples:", num_test_samples)

raw_data = np.array(raw_data[:])

mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std

print(raw_data)

sampling_rate = 6
sequence_length = 120
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 256

train_dataset = keras.utils.timeseries_dataset_from_array(
	raw_data[:-delay],
	targets=temperature[delay:],
	sampling_rate=sampling_rate,
	sequence_length=sequence_length,
	shuffle=True,
	batch_size=batch_size,
	start_index=0,
	end_index=num_train_samples)
	
print(train_dataset)

val_dataset = keras.utils.timeseries_dataset_from_array(
	raw_data[:-delay],
	targets=temperature[delay:],
	sampling_rate=sampling_rate,
	sequence_length=sequence_length,
	shuffle=True,
	batch_size=batch_size,
	start_index=num_train_samples,
	end_index=num_train_samples + num_val_samples)

test_dataset = keras.utils.timeseries_dataset_from_array(
	raw_data[:-delay],
	targets=temperature[delay:],
	sampling_rate=sampling_rate,
	sequence_length=sequence_length,
	shuffle=True,
	batch_size=batch_size,
	start_index=num_train_samples + num_val_samples)
	
for samples, targets in train_dataset.take(1):
	print("samples shape:", samples.shape)
	print("targets shape:", targets.shape)
	
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.Bidirectional(layers.LSTM(16))(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=val_dataset)	
					
loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training MAE")
plt.plot(epochs, val_loss, "b", label="Validation MAE")
plt.title("Training and validation MAE")
plt.legend()
plt.show()