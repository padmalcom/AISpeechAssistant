from loguru import logger
from chatbot import register_call
import global_variables
import yaml
import random
import os
from pygame import mixer
import text2numde

# Konfiguration muss in jedem Call neu gelesen werden, da beim Laden des gesamten Moduls
# die global_variables.voice_assistant noch nicht gesetzt ist und somit die Sprache nicht
# gelesen werden kann, da die Konfigurationsdatei global gelesen wird.
def __read_config__():
	cfg = None
		
	# Holen der Sprache aus der globalen Konfigurationsdatei
	LANGUAGE = global_variables.voice_assistant.cfg['assistant']['language']
		
	# Laden der intent-eigenen Konfigurationsdatei
	config_path = os.path.join('intents','functions','volume','config_volume.yml')
	with open(config_path, "r", encoding='utf8') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
	return cfg, LANGUAGE

@register_call("setVolume")
def setVolume(session_id = "general", volume=None):
	cfg, language = __read_config__()

	# konvertiere das Zahlenwort in einen geladenanzzahligen Wert
	if isinstance(volume, str):
		volume = text2numde.text2num(volume.strip())

	num_vol = volume
	

	# Konnte die Konfigurationsdatei des Intents geladen werden?
	if cfg:
	
		if num_vol < 0 or num_vol > 10:
			return random.choice(cfg['intent']['volume'][language]['invalid_volume'])
		else:
			new_volume = round(num_vol / 10.0, 1)
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
		
		new_volume = round(min(1.0, (vol + vol_up) / 10.0), 1)
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
		
		new_volume = round(max(0.0, (vol - vol_down) / 10.0), 1)
		logger.debug("Setze Lautstärke auf {}.", new_volume)
		global_variables.voice_assistant.tts.set_volume(new_volume)
		mixer.music.set_volume(new_volume)
		global_variables.voice_assistant.volume = new_volume
		return ""
	else:
		logger.error("Konnte Konfigurationsdatei für Intent 'volume' nicht laden.")
		return ""
