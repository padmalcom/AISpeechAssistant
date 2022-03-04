import numpy as np

def levenshtein_distance(source, target):
	rows = len(source)+1
	cols = len(target)+1
	distance = np.zeros((rows,cols),dtype = int)
	
	print(distance)

	# Initialisiere die erste Spalte und erste Zeile der Matrix
	for i in range(1, rows):
		for k in range(1,cols):
			distance[i][0] = i
			distance[0][k] = k
	print("")		
	print(distance)

	# Iteriere Zeilenweise über die Matrix
	for col in range(1, cols):
		for row in range(1, rows):
		
			# Stimmen die Zeichen überein? Dann
			if source[row-1] == target[col-1]:
				cost = 0
			else:
				cost = 1
			distance[row][col] = min(distance[row-1][col] + 1, # Kosten für Löschung
								 distance[row][col-1] + 1, # Kosten für Einfügen
								 distance[row-1][col-1] + cost) # Kosten für Ersetzen
	
	
	ratio = ((len(source)+len(target)) - distance[row][col]) / (len(source)+len(target))
	print("")
	print(distance) 
	
	return "Es werden {} Änderungen benötigt, um aus {} {} zu machen. Die Zeichenketten stimmen zu {}% überein.".format(distance[row][col], source, target, ratio*100)
		
if __name__ == '__main__':
	print("")
	print(levenshtein_distance("Pferd", "Herde"))