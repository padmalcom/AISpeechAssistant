from loguru import logger
from chatbot import register_call
import global_variables
import yaml
import random
import os
import text2numde
import constants

# Konfiguration muss in jedem Call neu gelesen werden, da beim Laden des gesamten Moduls
# die global_variables.voice_assistant noch nicht gesetzt ist und somit die Sprache nicht
# gelesen werden kann, da die Konfigurationsdatei global gelesen wird.
def __read_config__():
	cfg = None
		
	# Holen der Sprache aus der globalen Konfigurationsdatei
	LANGUAGE = global_variables.voice_assistant.cfg['assistant']['language']
		
	# Laden der intent-eigenen Konfigurationsdatei
	config_path = constants.find_data_file(os.path.join('intents','functions','volume','config_volume.yml'))
	with open(config_path, "r", encoding='utf-8') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
	return cfg, LANGUAGE

@register_call("getVolume")
def getVolume(session_id = "general", dummy=0):
	cfg, language = __read_config__()
	logger.info("Lautstärke ist {} von zehn.", int(global_variables.voice_assistant.volume * 10))
	return cfg['intent']['volume'][language]['volume_is'].format(int(global_variables.voice_assistant.volume * 10))

@register_call("setVolume")
def setVolume(session_id = "general", volume=None):
	cfg, language = __read_config__()
	
	if volume.strip() == "":
		return getVolume(session_id, 0)

	# konvertiere das Zahlenwort in einen geladenanzzahligen Wert
	if isinstance(volume, str):
		try:
			volume = text2numde.text2num(volume.strip())
		except:
			return random.choice(cfg['intent']['volume'][language]['invalid_volume'])
	num_vol = volume
		
	# Konnte die Konfigurationsdatei des Intents geladen werden?
	if cfg:
	
		if num_vol < 0 or num_vol > 10:
			logger.info("Lautstärke {} ist ungültig, nur Werte von 0 - 10 sind erlaubt.", num_vol)
			return random.choice(cfg['intent']['volume'][language]['invalid_volume'])
		else:
			new_volume = round(num_vol / 10.0, 1)
			logger.info("Setze Lautstärke von {} auf {}.", global_variables.voice_assistant.volume, new_volume) 
			global_variables.voice_assistant.tts.set_volume(new_volume)
			mixer.music.set_volume(new_volume)
			global_variables.voice_assistant.num_vol = new_volume
			return ""
	else:
		logger.error("Konnte Konfigurationsdatei für Intent 'volume' nicht laden.")
		return ""

@register_call("volumeUp")
def volumeUp(session_id = "general", volume=None):
	cfg, language = __read_config__()	

	# konvertiere das Zahlenwort in einen geladenanzzahligen Wert
	vol_up = 1
	if cfg:

		if isinstance(volume, str):
			vol_up = 1 + volume.split().count(cfg['intent']['volume'][language]['volume_up']) # Erlaube etwas wie "lauter, lauter, lauter"
		
		vol = global_variables.voice_assistant.volume
				
		new_volume = round(min(1.0, (vol + vol_up / 10.0)), 1)
		logger.info("Setze Lautstärke von {} auf {}.", global_variables.voice_assistant.volume, new_volume)
		logger.debug("Setze Lautstärke auf {}.", new_volume)
		global_variables.voice_assistant.tts.set_volume(new_volume)
		mixer.music.set_volume(new_volume)
		global_variables.voice_assistant.volume = new_volume
		return ""
	else:
		logger.error("Konnte Konfigurationsdatei für Intent 'volume' nicht laden.")
		return ""	
	
@register_call("volumeDown")
def volumeDown(session_id = "general", volume=None):
	cfg, language = __read_config__()
	
	vol_down = 1
	if cfg:
	
		if isinstance(volume, str):
			if text2numde.is_number(volume):
				vol_down = text2numde.text2num(volume.strip())
			else:
				vol_down = 1 + volume.split().count(cfg['intent']['volume'][language]['volume_down']) # Erlaube etwas wie "lauter, lauter, lauter"
		
		vol = global_variables.voice_assistant.volume
				
		new_volume = round(max(0.0, (vol - vol_down / 10.0)), 1)
		logger.info("Setze Lautstärke von {} auf {}.", global_variables.voice_assistant.volume, new_volume)
		logger.debug("Setze Lautstärke auf {}.", new_volume)
		global_variables.voice_assistant.tts.set_volume(new_volume)
		mixer.music.set_volume(new_volume)
		global_variables.voice_assistant.volume = new_volume
		return ""
	else:
		logger.error("Konnte Konfigurationsdatei für Intent 'volume' nicht laden.")
		return ""
