from datetime import datetime
import pytz
import global_variables
import os
import random
import yaml
import constants

def gettime(place="default"):

	
	config_path = constants.find_data_file(os.path.join('intents','functions','gettime','config_gettime.yml'))
	cfg = None
	with open(config_path, "r", encoding='utf-8') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
	
	if not cfg:
		logger.error("Konnte Konfigurationsdatei für gettime nicht lesen.")
		return ""
		
	# Holen der Sprache aus der globalen Konfigurationsdatei
	LANGUAGE = global_variables.voice_assistant.cfg['assistant']['language']
	
	# Der Ort ist nicht bekannt
	PLACE_UNKNOWN = random.choice(cfg['intent']['gettime'][LANGUAGE]['place_not_found'])
	
	# Wir fügen den unbekannten Ort in die Antwort ein
	PLACE_UNKNOWN = PLACE_UNKNOWN.format(place)

	# Lesen aller Orte aus der Konfigurationsdatei
	country_timezone_map = {}
	for key, value in cfg['intent']['gettime']['timezones'].items():
		country_timezone_map[key] = value

	# Versuche den angefragten Ort einer Zeitzone zuzuordnen
	timezone = None
	now = datetime.now()
	for c in country_timezone_map:
		if place.strip().lower() in country_timezone_map[c]:
			timezone = pytz.timezone(c)
			break
	
	# Wurde eine Zeitzone gefunden?
	if timezone:
		now = datetime.now(timezone)
		TIME_AT_PLACE = random.choice(cfg['intent']['gettime'][LANGUAGE]['time_in_place'])
		TIME_AT_PLACE = TIME_AT_PLACE.format(str(now.hour), str(now.minute), place.capitalize())
		return TIME_AT_PLACE
	else:
		# Falls nicht, prüfe, ob nach der Uhrzeit am Platz des Benutzers gefragt wurde
		if place == "default":
			TIME_HERE = random.choice(cfg['intent']['gettime'][LANGUAGE]['time_here'])
			TIME_HERE = TIME_HERE.format(str(now.hour), str(now.minute))
			return TIME_HERE
	
	# Wurde ein Ort angefragt, der nicht gefunden wurde, dann antworte entsprechend
	return PLACE_UNKNOWN