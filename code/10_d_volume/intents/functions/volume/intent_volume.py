from loguru import logger
from chatbot import register_call
import global_variables
import yaml
import random
import os
from pygame import mixer

# Konfiguration muss in jedem Call neu gelesen werden, da beim Laden des gesamten Moduls
# die global_variables.voice_assistant noch nicht gesetzt ist und somit die Sprache nicht
# gelesen werden kann, falls die Konfigurationsdatei global gelesen wird.
def __read_config__():
	cfg = None
		
	# Holen der Sprache aus der globalen Konfigurationsdatei
	LANGUAGE = global_variables.voice_assistant.cfg['assistant']['language']
		
	# Laden der intent-eigenen Konfigurationsdatei
	config_path = os.path.join('intents','functions','volume','config_volume.yml')
	with open(config_path, "r", encoding='utf8') as ymlfile:
		cfg, language = yaml.load(ymlfile, Loader=yaml.FullLoader)
	return cfg, language

@register_call("setvolume")
def setvolume(session_id = "general", volume=0):
	cfg, language = __read_config__()

	# Konnte die Konfigurationsdatei des Intents geladen werden?
	if cfg:
	
		if volume < 0 or volume > 10:
			return random.choice(cfg['intent']['volume'][language]['invalid_volume'])
		else:
			global_variables.voice_assistant.tts.set_volume(volume / 10.0)
			mixer.music.set_volume(volume / 10.0)
			global_variables.voice_assistant.volume = volume / 10.0
			return ""
	else:
		logger.error("Konnte Konfigurationsdatei für Intent 'volume' nicht laden.")
		return ""

@register_call("volumeup")
def volumeup(session_id = "general", volume=None):
	cfg, language = __read_config__()	
	if cfg:
		vol = global_variables.voice_assistant.volume
		volume_up_factor = 1 + param.split().count(cfg['intent']['volume'][language]['volume_up']) # Erlaube etwas wie "lauter, lauter, lauter"
		new_volume = min(1.0, (vol + volume_up_factor) / 10.0)
		logger.debug("Setze Lautstärke auf {}.", new_volume)
		global_variables.voice_assistant.tts.set_volume(new_volume)
		mixer.music.set_volume(new_volume)
		global_variables.voice_assistant.volume = new_volume
	else:
		logger.error("Konnte Konfigurationsdatei für Intent 'volume' nicht laden.")
		return ""	
	
@register_call("volumedown")
def volumedown(session_id = "general", volume=None):
	cfg, language = __read_config__()
	print("1")
	if cfg:
		print("2")
		vol = global_variables.voice_assistant.volume
		volume_up_factor = 1 + param.split().count(cfg['intent']['volume'][language]['volume_down']) # "leiser, leiser, leiser"
		new_volume = max(0.0, (vol + volume_up_factor) / 10.0)
		logger.debug("Setze Lautstärke auf {}.", new_volume)
		global_variables.voice_assistant.tts.set_volume(new_volume)
		mixer.music.set_volume(new_volume)
		global_variables.voice_assistant.volume = new_volume
	else:
		logger.error("Konnte Konfigurationsdatei für Intent 'volume' nicht laden.")
		return ""
