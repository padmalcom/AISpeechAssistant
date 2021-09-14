import librosa
from librosa import display
import scipy
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

if __name__ == "__main__":

	# Laden der Audiodatei
	file_path = "test.wav"
	samples, sampling_rate = librosa.load(file_path)
	print("# Samples: " + str(len(samples)) + ", Sampling Rate: " + str(sampling_rate))
	
	# Berechnen der Audiolänge
	duration = len(samples) / sampling_rate
	print("Audiolänge: " + str(duration))
	
	# Darstellung der Amplitude über die Zeit
	plt.figure()
	librosa.display.waveplot(y = samples, sr = sampling_rate)
	plt.xlabel("Zeit (s)")
	plt.ylabel("Amplitude (db)")
	plt.show()
	
	# FFT mit Scipy
	n = len(samples)
	T = 1 / sampling_rate
	yf = scipy.fft.fft(samples)
	xf = np.linspace(0.0, 1.0 / (2.0*T), n // 2)
	fig, ax = plt.subplots()
	ax.plot(xf, 2.0/n * np.abs(yf[:n//2]))
	plt.grid()
	plt.xlabel("Frequenz")
	plt.ylabel("Magnitude")
	plt.show()
	
	# Ableiten des Spectrogramms über Scipy
	frequencies, times, spectrogram = signal.spectrogram(samples, sampling_rate)
	plt.specgram(samples,Fs=sampling_rate)
	plt.title('Spectrogram')
	plt.ylabel('Frequenzband')
	plt.xlabel('Zeitfenster')
	plt.show()