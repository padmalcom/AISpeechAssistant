from loguru import logger
import yaml
import time
import pvporcupine
import pyaudio
import struct
import os
import sys
from vosk import Model, SpkModel, KaldiRecognizer
import json
import tinydb
import numpy as np
from usermgmt import UserMgmt
import io

from pygame import mixer

from TTS import Voice
import multiprocessing

from intentmgmt import IntentMgmt

CONFIG_FILE = "config.yml"

#va = None

class VoiceAssistant():

	def __init__(self):
		logger.info("Initialisiere VoiceAssistant...")
		
		logger.debug("Lese Konfiguration...")
		
		global CONFIG_FILE
		with open(CONFIG_FILE, "r", encoding='utf8') as ymlfile:
			self.cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
		if self.cfg:
			logger.debug("Konfiguration gelesen.")
		else:
			logger.debug("Konfiguration konnte nicht gelesen werden.")
			sys.exit(1)
		language = self.cfg['assistant']['language']
		if not language:
			language = "German"
		logger.info("Verwende Sprache {}", language)
			
		logger.debug("Initialisiere Wake Word Erkennung...")
		self.wake_words = self.cfg['assistant']['wakewords']
		if not self.wake_words:
			self.wake_words = ['bumblebee']
		logger.debug("Wake words are {}", ','.join(self.wake_words))
		self.porcupine = pvporcupine.create(keywords=self.wake_words)
		logger.debug("Wake Word Erkennung wurde initialisiert.")
		
		logger.debug("Initialisiere Audioeingabe...")
		self.pa = pyaudio.PyAudio()
		
		self.audio_stream = self.pa.open(
			rate=self.porcupine.sample_rate,
			channels=1,
			format=pyaudio.paInt16,
			input=True,
			frames_per_buffer=self.porcupine.frame_length,
			input_device_index=0)
		logger.debug("Audiostream geöffnet.")

		logger.info("Initialisiere Sprachausgabe...")
		self.tts = Voice()
		voices = self.tts.get_voice_keys_by_language(language)
		if len(voices) > 0:
			logger.info('Stimme {} gesetzt.', voices[0])
			self.tts.set_voice(voices[0])
		else:
			logger.warning("Es wurden keine Stimmen gefunden.")
		self.tts.say("Sprachausgabe aktiviert.")
		logger.debug("Sprachausgabe initialisiert")
		
		logger.info("Initialisiere Spracherkennung...")
		stt_model = Model('./vosk-model-de-0.6')
		speaker_model = SpkModel('./vosk-model-spk-0.4')
		self.rec = KaldiRecognizer(stt_model, speaker_model, 16000)
		self.is_listening = False
		logger.info("Initialisierung der Spracherkennung abgeschlossen.")
		
		logger.info("Initialisiere Benutzerverwaltung...")
		self.user_management = UserMgmt(init_dummies=True)
		self.allow_only_known_speakers = self.cfg["assistant"]["allow_only_known_speakers"]
		logger.info("Benutzerverwaltung initialisiert")
		
		# Initialisiere den Audio-Player
		mixer.init()
		
		logger.info("Initialisiere Intent-Management...")
		self.intent_management = IntentMgmt()
		logger.info('{} intents geladen', self.intent_management.get_count())
		self.tts.say("Initialisierung abgeschlossen")
	
	# Finde den besten Sprecher aus der Liste aller bekannter Sprecher aus dem User Management
	def __detectSpeaker__(self, input):
		bestSpeaker = None
		bestCosDist = 100
		for speaker in self.user_management.speaker_table.all():
			nx = np.array(speaker.get('voice'))
			ny = np.array(input)
			cosDist = 1 - np.dot(nx, ny) / np.linalg.norm(nx) / np.linalg.norm(ny)
			if (cosDist < bestCosDist):
				if (cosDist < 0.3):
					bestCosDist = cosDist
					bestSpeaker = speaker.get('name')
		return bestSpeaker		
			
	def run(self):
		logger.info("VoiceAssistant Instanz wurde gestartet.")

if __name__ == '__main__':
	import global_variables
	multiprocessing.set_start_method('spawn')
	global_variables.voice_assistant = VoiceAssistant()
	logger.info("Anwendung wurde gestartet")
	global_variables.voice_assistant.run()
		
	try:
		while True:
		
			pcm = global_variables.voice_assistant.audio_stream.read(global_variables.voice_assistant.porcupine.frame_length)
			pcm_unpacked = struct.unpack_from("h" * global_variables.voice_assistant.porcupine.frame_length, pcm)		
			keyword_index = global_variables.voice_assistant.porcupine.process(pcm_unpacked)
			if keyword_index >= 0:
				logger.info("Wake Word {} wurde verstanden.", global_variables.voice_assistant.wake_words[keyword_index])
				global_variables.voice_assistant.is_listening = True
				
			# Spracherkennung
			if global_variables.voice_assistant.is_listening:
				if global_variables.voice_assistant.rec.AcceptWaveform(pcm):
					recResult = json.loads(global_variables.voice_assistant.rec.Result())
					
					speaker = global_variables.voice_assistant.__detectSpeaker__(recResult['spk'])
					if (speaker == None) and (global_variables.voice_assistant.allow_only_known_speakers == True):
						global_variables.voice_assistant.tts.say("Ich kenne deine Stimme nicht und darf damit keine Befehle von dir entgegen nehmen.")
						global_variables.voice_assistant.current_speaker = None
					else:
						if speaker:
							logger.debug("Sprecher ist {}", speaker)
						global_variables.voice_assistant.current_speaker = speaker
						global_variables.voice_assistant.current_speaker_fingerprint = recResult['spk']
						logger.debug('Ich habe verstanden "{}"', recResult['text'])
						
						# Lasse den Assistenten auf die Spracheingabe reagieren
						output = global_variables.voice_assistant.intent_management.process(recResult['text'], speaker)
						global_variables.voice_assistant.tts.say(output)
						
						global_variables.voice_assistant.is_listening = False
						global_variables.voice_assistant.current_speaker = None
				
	except KeyboardInterrupt:
		logger.debug("Per Keyboard beendet")
	finally:
		logger.debug('Beginne Aufräumarbeiten...')
		if global_variables.voice_assistant.porcupine:
			global_variables.voice_assistant.porcupine.delete()
			
		if global_variables.voice_assistant.audio_stream is not None:
			global_variables.voice_assistant.audio_stream.close()
			
		if global_variables.voice_assistant.pa is not None:
			global_variables.voice_assistant.pa.terminate()			