from loguru import logger
from chatbot import register_call
import global_variables
import yaml
import random
import os
from pygame import mixer

# Spezieller Intent, der Zugriff auf voice_assistant braucht	
@register_call("stop")
def stop(session_id = "general", dummy=0):

	cfg = None
	
	# Laden der intent-eigenen Konfigurationsdatei
	config_path = os.path.join('intents','functions','stop','config_stop.yml')
	with open(config_path, "r", encoding='utf8') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
	
	# Holen der Sprache aus der globalen Konfigurationsdatei
	LANGUAGE = global_variables.voice_assistant.cfg['assistant']['language']
	
	# Konnte die Konfigurationsdatei des Intents geladen werden?
	if cfg:
	
		# Setze einen Defaultwert für die Rückgabe, falls der Assistent derzeit nicht spricht
		result = random.choice(cfg['intent']['stop'][LANGUAGE]['not_saying_anything'])
		
		# Spricht er?
		if global_variables.voice_assistant.tts.is_busy():
			global_variables.voice_assistant.tts.stop()
			result = random.choice(cfg['intent']['stop'][LANGUAGE]['be_silent'])
			
		# Wird ein Sound ausgegeben? Stoppe ihn
		if mixer.music.get_busy():
			mixer.music.stop()
			result = random.choice(cfg['intent']['stop'][LANGUAGE]['be_silent'])
			
		return result
	else:
		logger.error("Konnte Konfigurationsdatei für Intent 'stop' nicht laden.")
		return ""