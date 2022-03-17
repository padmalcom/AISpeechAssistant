import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np

df = pd.read_csv('weather_data.csv')

titles = [
    "pressure_air_site",
    "temperature_air_mean_200",
    "temperature_air_mean_005",
    "humidity",
    "temperature_dew_point_mean_200"
]

feature_keys = [
    "pressure_air_site",
    "temperature_air_mean_200",
    "temperature_air_mean_005",
    "humidity",
    "temperature_dew_point_mean_200"
]

feature_keys_old = [
    "p (mbar)",
    "T (K)",
	"T (K)",
    "rh (%)",
    "T (K)"
]

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple"
]

date_time_key = "date"


def show_raw_visualization(data):
    time_data = data[date_time_key]
    fig, axes = plt.subplots(
        nrows=7, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title="{} - {}".format(titles[i], key),
            rot=25,
        )
        ax.legend([titles[i]])
    plt.tight_layout()


#show_raw_visualization(df)

def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()


#show_heatmap(df)


split_fraction = 0.715
train_split = int(split_fraction * int(df.shape[0]))
step = 6

past = 720
future = 72
learning_rate = 0.001
batch_size = 256
epochs = 1


def standardize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std, data_mean, data_std
	
def inverse_standardization(value, mean, std):
	result = value * std + mean
	print("processing:", value, "res:", result)
	return result[2]# keep only temperature
	
def kelvin_to_celsius(value):
	return value - 273.15

print(
    "The selected parameters are:",
    ", ".join([titles[i] for i in [1, 3, 4]]),
)
selected_features = [feature_keys[i] for i in [1, 3, 4]]
features = df[selected_features]
features.index = df[date_time_key]
print("0", features.head())

features, mean, std = standardize(features.values, train_split)
features = pd.DataFrame(features)
print("1", features.head())

train_data = features.loc[0 : train_split - 1]
val_data = features.loc[train_split:]


start = past + future
end = start + train_split

print(train_data.head())

x_train = train_data[[i for i in range(3)]].values
y_train = features.iloc[start:end][[2]]

print("xtrain", x_train)
print("ytrain", y_train)

sequence_length = int(past / step)


dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

x_end = len(val_data) - past - future

label_start = train_split + past + future

x_val = val_data.iloc[:x_end][[i for i in range(3)]].values
y_val = features.iloc[label_start:][[2]]

dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    x_val,
    y_val,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)


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


def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


visualize_loss(history, "Training and Validation Loss")


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
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    plt.show()
    return


for x, y in dataset_val.take(5):

    history = x[0][:, 1].numpy()
    ground_truth = y[0].numpy()
    prediction = model.predict(x)[0]
		
    history = np.array([kelvin_to_celsius(inverse_standardization(xi, mean, std)) for xi in history])
    ground_truth = np.array([kelvin_to_celsius(inverse_standardization(xi, mean, std)) for xi in ground_truth])
    prediction = np.array([kelvin_to_celsius(inverse_standardization(xi, mean, std)) for xi in prediction])
	
    print("history:", history)
    print("ground_truth:", ground_truth)
    print("prediction:", prediction)
	
    #history = history[:,2]
	#ground_truth = ground_truth[:,2]
	
	
    show_plot(
        [history, ground_truth, prediction],
        12,
        "Single Step Prediction",
    )


