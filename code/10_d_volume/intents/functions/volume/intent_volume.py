from loguru import logger
from chatbot import register_call
import global_variables
import yaml
import random
import os
from pygame import mixer

@register_call("set_volume")
def set_volume(session_id = "general", volume=0):

	cfg = None
	
	# Laden der intent-eigenen Konfigurationsdatei
	config_path = os.path.join('intents','functions','stop','config_stop.yml')
	with open(config_path, "r", encoding='utf8') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
	
	# Holen der Sprache aus der globalen Konfigurationsdatei
	LANGUAGE = global_variables.voice_assistant.cfg['assistant']['language']
	
	# Konnte die Konfigurationsdatei des Intents geladen werden?
	if cfg:
	
		if volume < 0 or volume > 10:
			return random.choice(cfg['intent']['volume'][LANGUAGE]['invalid_volume'])
		else:
			global_variables.voice_assistant.tts.set_volume(volume / 10.0)
			mixer.music.set_volume(volume / 10.0)
			global_variables.voice_assistant.volume = volume / 10.0
			return ""
	else:
		logger.error("Konnte Konfigurationsdatei für Intent 'volume' nicht laden.")
		return ""

@register_call("volume_up")
def set_volume(session_id = "general", param=""):
	vol = global_variables.voice_assistant.volume
	volume_up_factor = 1 + param.split().count("lauter") # Erlaube etwas wie "lauter, lauter, lauter"
	global_variables.voice_assistant.tts.set_volume(min(1.0, (vol + volume_up_factor) / 10.0))
	mixer.music.set_volume(min(1.0, (vol + volume_up_factor) / 10.0))
	global_variables.voice_assistant.volume = min(1.0, (vol + volume_up_factor) / 10.0)
	
@register_call("volume_down")
def set_volume(session_id = "general", param=""):
	vol = global_variables.voice_assistant.volume
	volume_up_factor = 1 + param.split().count("leiser") # "leiser, leiser, leiser"
	global_variables.voice_assistant.tts.set_volume(max(0.0, (vol + volume_up_factor) / 10.0))
	mixer.music.set_volume(max(0.0, (vol + volume_up_factor) / 10.0))
	global_variables.voice_assistant.volume = max(0.0, (vol + volume_up_factor) / 10.0)	
