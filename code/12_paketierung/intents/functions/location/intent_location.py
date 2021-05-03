from loguru import logger
from chatbot import register_call
import global_variables
import random
import os
import yaml
import geocoder
import constants

@register_call("location")
def location(session_id = "general", dummy=0):

	config_path = constants.find_data_file(os.path.join('intents','functions','location','config_location.yml'))
	cfg = None
	
	with open(config_path, "r", encoding='utf-8') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
	
	if not cfg:
		logger.error("Konnte Konfigurationsdatei für die Lokalisierung nicht lesen.")
		return ""
		
	# Holen der Sprache aus der globalen Konfigurationsdatei
	LANGUAGE = global_variables.voice_assistant.cfg['assistant']['language']
	
	YOU_ARE_HERE = random.choice(cfg['intent']['location'][LANGUAGE]['youarehere'])
	
	# Ermittle den Standort mittels IP
	loc = geocoder.ip('me')
	logger.debug("Random template {} and city {}", YOU_ARE_HERE, loc.city)
	return YOU_ARE_HERE.format(loc.city)