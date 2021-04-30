from loguru import logger
from chatbot import register_call
import global_variables
import random
import os
import yaml
import requests
import constants

@register_call("smarthome")
def smarthome(session_id = "general", device="", state=""):

	config_path = constants.find_data_file(os.path.join('intents','functions','smarthome','config_smarthome.yml'))
	cfg = None
	
	with open(config_path, "r", encoding='utf-8') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
	
	if not cfg:
		logger.error("Konnte Konfigurationsdatei für den Intent 'smarthome' nicht lesen.")
		return ""

	LANGUAGE = global_variables.voice_assistant.cfg['assistant']['language']
	# Lese alle bekannten Smart Home Devices aus
	devices = {}
	for key, value in cfg['intent']['smarthome']['devices'].items():
		devices[key] = value
	
	# Gibt es das angefragte Device?
	if devices[device]:	
		s = None
		# Interpretiere den Zustand
		if state in cfg['intent']['smarthome'][LANGUAGE]['state_on']:
			s = "on"
		elif state in cfg['intent']['smarthome'][LANGUAGE]['state_off']:
			s = "off"
		else:
			return cfg['intent']['smarthome'][LANGUAGE]['state_unknown'].format(state)
			
		# Setze einen Get-Request ab, der das Gerät ein- oder ausschaltet
		PARAMS = {'turn': s}

		try:
			r = requests.get(url = "http://" + devices[device] + "/relay/0", params = PARAMS)
		except:
			return cfg['intent']['smarthome'][LANGUAGE]['request_failed'].format(device)

		data = r.json()
		logger.debug(data)
		return ""
	return cfg['intent']['smarthome'][LANGUAGE]['device_unknown'].format(device)