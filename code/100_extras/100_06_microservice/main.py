import requests

intents = {
	'gettime': 'http://192.168.1.151:5000/gettime/<place>'
}

if __name__ == '__main__':

	# Annahme: Intent Management hat gettime als passenden Intent ermittelt und die Slots befüllt.
	intent = 'gettime'
	slots = {
		'place': 'Deutschland'
	}
	
	# Ermittle den Microservice
	endpoint = ""
	for key in intents:
		if key == intent:
			endpoint = intents[key]
			break
			
	# Befülle die Slots indem die Variablen in <> aus der Service-URL ersetzt werden
	for slot in slots:
		endpoint = endpoint.replace('<' + slot + '>', slots[slot])
		
	response = requests.get(endpoint)

	output = ""
	if response.status_code == 200:
		output = response.json()
	elif response.status_code == 404:
		print('Der Service ' + intent + ' ist nicht erreichbar.')

	print(output)
	
	
