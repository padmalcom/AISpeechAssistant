from chatbot import register_call
import global_variables
import random
import os
import yaml

@register_call("animalSound")
def animalSound(session_id = "general", animal="none"):

	config_path = os.path.join('intents','functions','animalsounds','config_animalsounds.yml')
	ogg_path = os.path.join('intents','functions','animalsounds','animals')
	cfg = None
	with open(config_path, "r", encoding='utf8') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
	
	if not cfg:
		logger.error("Konnte Konfigurationsdatei für animalsounds nicht lesen.")
		return ""
		
	# Holen der Sprache aus der globalen Konfigurationsdatei
	LANGUAGE = global_variables.voice_assistant.cfg['assistant']['language']
	
	# Das Tier ist nicht bekannt
	ANIMAL_UNKNOWN = random.choice(cfg['intent']['animalsounds'][LANGUAGE]['animal_not_found'])
	
	animals = {}
	for key, value in cfg['intent']['animalsounds']['animals'].items():
		animals[key] = value

	for a in animals:
		if animal.strip().lower() in animals[a]:
			ogg_file = os.path.join(ogg_path, a + '.ogg')
			#if mixer.music.get_busy():
			#	mixer.music.stop()
			#mixer.music.load(ogg_file)
			#mixer.music.play()
			global_variables.voice_assistant.audio_player.play_file(ogg_file)
			
			# Der Assistent muss nicht sprechen, wenn ein Tierlaut wiedergegeben wird
			return ""
	
	# Wurde kein Tier gefunden?
	return ANIMAL_UNKNOWN