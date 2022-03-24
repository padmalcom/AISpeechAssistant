import numpy as np
from sklearn.linear_model import LinearRegression

# x
x1 = np.array([[0], [1], [2], [3], [4], [5]]) # x must be two dimensional
y1 = np.array([0, 1, 2, 3, 4, 5])
val_x1 = np.array([[6], [7], [8]]) # x must be two dimensional
val_y1 = np.array([6, 7, 8])

# x square
x2 = np.array([[0], [1], [2], [3], [4], [5]])
y2 = np.array([0, 1, 4, 9, 16, 25])
val_x2 = np.array([[6], [7], [8]])
val_y2 = np.array([36, 49, 64])

# fibonacchi
x3 = np.array([[0], [1], [2], [3], [4], [5]])
y3 = np.array([0, 1, 1, 2, 3, 5])
val_x3 = np.array([[6], [7], [8]])
val_y3 = np.array([8, 13, 21])

test = np.array([[6], [7], [8]])

def train_and_predict(x, y, val_x, val_y, test):
	model = LinearRegression()
	model.fit(x, y)

	cod = model.score(val_x, val_y)
	print('Coefficient of determination: ', cod)

	predictions = model.predict(test)
	print('Predictions: ', predictions, sep='\n')
	
if __name__ == '__main__':
	print("x")
	train_and_predict(x1, y1, val_x1, val_y1, test)
	print("\n")
	
	print("Square")
	train_and_predict(x2, y2, val_x2, val_y2, test)
	print("\n")
	
	print("Fibonacci")
	train_and_predict(x3, y3, val_x3, val_y3, test)
	print("\n")