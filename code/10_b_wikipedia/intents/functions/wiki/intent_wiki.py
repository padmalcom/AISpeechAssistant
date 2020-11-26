from loguru import logger
from chatbot import register_call
import global_variables
import yaml
import random
import os
import wikipedia
import pycountry

@register_call("wiki")
def wiki(session_id = "general", query="none"):
	cfg = None
	
	# Laden der intent-eigenen Konfigurationsdatei
	config_path = os.path.join('intents','functions','wiki','config_wiki.yml')
	with open(config_path, "r", encoding='utf8') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
	
	# Holen der Sprache aus der globalen Konfigurationsdatei
	LANGUAGE = global_variables.voice_assistant.cfg['assistant']['language']
	
	# Setze die richtige Sprache für Wikipedia
	lang = pycountry.languages.get(name=LANGUAGE) # Hole den iso-638 code für die Sprache
	if lang:
		wikipedia.set_lang(lang.alpha_2)

	UNKNOWN_ENTITY = random.choice(cfg['intent']['wiki'][LANGUAGE]['unknown_entity'])
	UNKNOWN_ENTITY = UNKNOWN_ENTITY.format(query)
	
	# Konnte die Konfigurationsdatei des Intents geladen werden?
	if cfg:
		query = query.strip()
		try:
			return wikipedia.summary(query, sentences=1)
		except Exception:
			for new_query in wikipedia.search(query):
				try:
					return wikipedia.summary(new_query)
				except Exception:
					pass
		return UNKNOWN_ENTITY
	else:
		logger.error("Konnte Konfigurationsdatei für Intent 'wikipedia' nicht laden.")
		return ""