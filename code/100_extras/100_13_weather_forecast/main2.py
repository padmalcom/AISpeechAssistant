import os
import csv
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tabulate import tabulate
import sys

fname = os.path.join("weather_data.csv")

temperature = []
raw_data = []

with open(fname, newline='') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',', quotechar='\'')
		
	for i, row in enumerate(spamreader):
		if i==0:
			continue
		row[2] = float(row[2]) - 273.15 # kelvin to celsius
		temperature.append(float(row[2]))
		raw_data.append([float(x) for x in row[2:6]])
		
print(raw_data[:5])
print(temperature[:5])

num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples
print("num_train_samples:", num_train_samples)
print("num_val_samples:", num_val_samples)
print("num_test_samples:", num_test_samples)

raw_data = np.array(raw_data[:])

mean = raw_data[:num_train_samples].mean(axis=0)
std = raw_data[:num_train_samples].std(axis=0)
raw_data = (raw_data - mean) / std

# new
temperature = [(x - mean[0]) / std[0] for x in temperature]


#print(raw_data)
#sys.exit()

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
	
print("features:", raw_data.shape[-1])
	
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

def show_plot(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, val in enumerate(plot_data):
        print(i)
        if i:
            if plot_data[i].ndim == 1: # true future
                plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
            else:           
                plt.plot(future, plot_data[i][0,0], marker[i], markersize=10, label=labels[i])
        else:
            #plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
            plt.plot(time_steps, [x[0] for x in plot_data[i]], marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    plt.show()
	
# evaluate

for x, y in val_dataset.take(1):

    print("x", x[0], "y", y[0])
    history = x[0][:, 0].numpy()
    ground_truth = y[0].numpy()
    prediction = model.predict(x)[0]

    history = np.array([xi * std  + mean for xi in history])
    ground_truth = np.array([ground_truth * std + mean])
    prediction = np.array([prediction * std + mean])
	
    print("history:", history)
    print("ground_truth:", ground_truth)
    print("prediction:", prediction)
	
    #history = history[:,2]
	#ground_truth = ground_truth[:,2]
	
	
    show_plot(
        [history, ground_truth, prediction],
        1, # 1 day in the future
        "Single Step Prediction",
    )

#inference
rd = np.asarray(raw_data[-120:])
rd = np.expand_dims(rd, axis=0) 
print("0", rd)
prediction = model.predict(rd)
print("1", prediction)
prediction = prediction[0]
print("4 Prediction test for x=", rd, " is: ", prediction)
