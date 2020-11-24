from chatbot import register_call
import global_variables
import random

@register_call("animalSound")
def animalSound(animal="none", session_id = "general"):

	with open("config_animalsounds.yml", "r") as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
		
	# Holen der Sprache aus der globalen Konfigurationsdatei
	LANGUAGE = global_variables.voice_assistant.cfg['assistant']['language']
	
	# Das Tier ist nicht bekannt
	ANIMAL_UNKNOWN = random.choice(cfg['intent']['animalsound'][LANGUAGE]['animal_not_found'])
	
	animals = {}
	for key, value in cfg['intent']['animalsounds']['animals'].items():
		animals[key] = value
	
	print(animals)
	
	for a in animals:
		if animal in animals[a]:
			mp3_path = os.path.join('animals', a + '.mp3')
			print("Playing " + str(mp3))
			if global_variables.voice_assistant.mixer.music.get_busy():
				global_variables.voice_assistant.mixer.music.stop()
			global_variables.voice_assistant.mixer.music.load(mp3_path)
			global_variables.voice_assistant.mixer.music.play()
			
			# Der Assistent muss nicht sprechen, wenn ein Tierlaut wiedergegeben wird
			return ""
	
	# Wurde kein Tier gefunden?
	return ANIMAL_UNKNOWN