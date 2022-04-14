import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# x
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]).reshape((-1, 1))
y = np.array([4.5, 4.7, 4.5, 4.4, 4.5, 4.4, 4.3, 4.4, 4.5, 4.6, 4.7, 4.7, 4.8, 4.7])
val_x = np.array([14, 15, 16]).reshape((-1, 1))
val_y = np.array([4.2, 4.3, 4.4])

test = np.array([17, 18, 19]).reshape((-1, 1))

def calc_slope_line(x, slope, intercept):
  return slope * x + intercept



def train_and_predict(x, y, val_x, val_y, test):

	model = LinearRegression()
	model.fit(x, y)

	cod = model.score(val_x, val_y)
	print('Coefficient of determination: ', cod)

	predictions = model.predict(test)
	print('Predictions: ', predictions, sep='\n')
	
	print("slope (Steigung):", model.coef_, "intercept (y-Achsenabschnitt):", model.intercept_)
	
	slope_line = []
	for v in x:
		slope_line.append(calc_slope_line(v, model.coef_, model.intercept_))
	plt.scatter(x, y, marker='x', color='r')
	plt.plot(x, slope_line)
	plt.scatter(test, predictions, marker='o', color='g')
	plt.title("Temperaturmessung der letzten 13 Tage")
	plt.xlabel("Letzte 13 Stunden")
	plt.ylabel("Temperatur in C")	
	plt.show()
	
if __name__ == '__main__':
	train_and_predict(x, y, val_x, val_y, test)