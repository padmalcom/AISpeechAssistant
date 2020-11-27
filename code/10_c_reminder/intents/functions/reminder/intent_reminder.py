from datetime import datetime
import pytz
import global_variables
import os
import random
import yaml

def reminder(time="", action_conj="", action_inf=""):

	# Hole den aktuellen Sprecher, falls eine persönliche Erinnerung stattfinden soll
	speaker = global_variables.voice_assistant.current_speaker
	
	config_path = os.path.join('intents','functions','reminder','config_reminder.yml')
	cfg = None
	with open(config_path, "r", encoding='utf8') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
	
	if not cfg:
		logger.error("Konnte Konfigurationsdatei für reminder nicht lesen.")
		return ""
		
	# Holen der Sprache aus der globalen Konfigurationsdatei
	LANGUAGE = global_variables.voice_assistant.cfg['assistant']['language']
	
	
	# Wurde eine Zeitzone gefunden?
	if timezone:
		now = datetime.now(timezone)
		TIME_AT_PLACE = random.choice(cfg['intent']['gettime'][LANGUAGE]['time_in_place'])
		TIME_AT_PLACE = TIME_AT_PLACE.format(str(now.hour), str(now.minute), country.capitalize())
		return TIME_AT_PLACE
	else:
		# Falls nicht, prüfe, ob nach der Uhrzeit am Platz des Benutzers gefragt wurde
		if country == "default":
			TIME_HERE = random.choice(cfg['intent']['gettime'][LANGUAGE]['time_here'])
			TIME_HERE = TIME_HERE.format(str(now.hour), str(now.minute))
			return TIME_HERE
	
	# Wurde ein Ort angefragt, der nicht gefunden wurde, dann antworte entsprechend
	return PLACE_UNKNOWN