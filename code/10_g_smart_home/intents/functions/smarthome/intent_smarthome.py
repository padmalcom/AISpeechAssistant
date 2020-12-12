from loguru import logger
from chatbot import register_call
import global_variables
import random
import os
import yaml
import requests

@register_call("smarthome")
def smarthome(session_id = "general", switch="", state=""):

	config_path = os.path.join('intents','functions','smarthome','config_smarthome.yml')
	cfg = None
	
	with open(config_path, "r", encoding='utf8') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
	
	if not cfg:
		logger.error("Konnte Konfigurationsdatei für das Smart Home nicht lesen.")
		return ""

	LANGUAGE = global_variables.voice_assistant.cfg['assistant']['language']
	
	# Lese alle bekannten Smart Home Devices aus
	devices = {}
	for key, value in cfg['intent']['smarthome']['devices'].items():
		devices[key] = value
		
	# Gibt es das angefragte Device?
	if devices[switch]:	
		s = None
		# Interpretiere den Zustand
		if state in cfg['intent']['smarthome'][LANGUAGE]['on']:
			s = "on"
		elif state in cfg['intent']['smarthome'][LANGUAGE]['off']:
			s = "off"
		else:
			return cfg['intent']['smarthome']['state_unknown']
			
		# Setze einen Get-Request ab, der das Gerät ein- oder ausschaltet
		PARAMS = {'turn', s}
		r = requests.get(url = "http://" + devices[switch] + "/relay/0", params = PARAMS) 
		data = r.json()
		logger.debug(data)
	else:
		return cfg['intent']['smarthome']['device_unknown']