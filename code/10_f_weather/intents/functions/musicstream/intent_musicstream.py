from chatbot import register_call
import global_variables
import random
import os
import yaml
from pygame import mixer

@register_call("playRadio")
def playRadio(session_id = "general", station=None):

	config_path = os.path.join('intents','functions','musicstream','config_musicstream.yml')
	cfg = None
	with open(config_path, "r", encoding='utf8') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
	
	if not cfg:
		logger.error("Konnte Konfigurationsdatei für das Musikstreaming nicht lesen.")
		return ""
		
	# Holen der Sprache aus der globalen Konfigurationsdatei
	LANGUAGE = global_variables.voice_assistant.cfg['assistant']['language']
	
	# Meldung falls der Sender nicht gefunden wurde
	UNKNOWN_STATION = random.choice(cfg['intent']['animalsounds'][LANGUAGE]['unknown_station'])
	
	stations = {}
	for key, value in cfg['intent']['musicstream']['stations'].items():
		stations[key] = value

	if stations[station]:
		if mixer.music.get_busy():
			mixer.music.stop()
		mixer.music.load(stations[station])
		mixer.music.play()
		
		# Der Assistent muss nicht sprechen, wenn ein Radiostream gespielt wird
		return ""
	
	# Wurde kein Sender gefunden?
	return UNKNOWN_STATION