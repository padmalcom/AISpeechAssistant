from loguru import logger
from chatbot import register_call
import global_variables
import yaml
import random

# Spezieller Intent, der Zugriff auf voice_assistant braucht	
@register_call("stop")
def stop(dummy=0, session_id = "general"):

	cfg = None
	
	# Laden der intent-eigenen Konfigurationsdatei
	with open("config_animalsounds.yml", "r") as ymlfile:
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
		if global_variables.voice_assistant.mixer.music.get_busy():
			global_variables.voice_assistant.mixer.music.stop()
			result = random.choice(cfg['intent']['stop'][LANGUAGE]['be_silent'])
			
		return result
	else:
		logger.error("Konnte Konfigurationsdatei für Intent 'stop' nicht laden.")
		return ""