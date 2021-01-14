from loguru import logger
from chatbot import register_call
import global_variables
import random
import os
import yaml

import text2numde
from fuzzywuzzy import fuzz

def musicstream(station=None):

	config_path = os.path.join('intents','functions','musicstream','config_musicstream.yml')
	cfg = None
	
	with open(config_path, "r", encoding='utf8') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
	
	if not cfg:
		logger.error("Konnte Konfigurationsdatei für das Musikstreaming nicht lesen.")
		return ""
		
	# Holen der Sprache aus der globalen Konfigurationsdatei
	LANGUAGE = global_variables.voice_assistant.cfg['assistant']['language']
	
	# Meldung falls der Sender nicht gefunden wurde
	UNKNOWN_STATION = random.choice(cfg['intent']['musicstream'][LANGUAGE]['unknown_station'])
	
	if (station == None) or (station == ""):
		return UNKNOWN_STATION
	
	# Radiosender haben häufig Zahlen in den Namen, weswegen wir für einen besseren Abgleich
	# Zahlenwörter in Zahlenwerte umwandeln.
	station = text2numde.sentence2num(station)
	
	# Wir eliminieren weiterhin alle Whitespaces, denn das Buchstabieren in VOSK bringt
	# pro Buchstabe eine Leerstelle mit sich.
	station = "".join(station.split())
	
	station_stream = None
	for key, value in cfg['intent']['musicstream']['stations'].items():
	
		# Wir führen eine Fuzzy-Suche aus, da die Namen der Radiosender nicht immer perfekt
		# von VOSK erkannt werden.
		ratio = fuzz.ratio(station.lower(), key.lower())
		logger.info("Übereinstimmung von {} und {} ist {}%", station, key, ratio)
		if ratio > 70:
			station_stream = value
			break

	# Wurde kein Sender gefunden?
	if station_stream is None:
		return UNKNOWN_STATION
		
	#if mixer.music.get_busy():
		#mixer.music.stop()
	#sound=mixer.Sound(station_stream)
	#sound.play()
	#mixer.music.load(station_stream)
	#mixer.music.play()
	#global_variables.voice_assistant.play_audio(station_stream)
	global_variables.voice_assistant.audio_player.play_stream(station_stream)
		
	# Der Assistent muss nicht sprechen, wenn ein Radiostream gespielt wird
	return ""	