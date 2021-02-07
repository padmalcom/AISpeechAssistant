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

from chatbot import Chat, register_call
from datetime import datetime

from TTS import Voice
import multiprocessing

CONFIG_FILE = "config.yml"

@register_call("time")
# dummy muss theoretisch nicht auf 0 gesetzt werden, Python fordert aber,
# dass ein Parameter, der einen Default-Wert erhält, nur von weiteren Parametern gefolgt
# werden dürfen, die auch einen Default-Wert haben
def getTime(session_id = "general", dummy=0):
	now = datetime.now()
	return "Es ist " + str(now.hour) + " Uhr und " + str(now.minute) + " Minuten."
	
@register_call("stop")
def stop(session_id = "general", dummy=0):
	if va.tts.is_busy():
		va.tts.stop()
		return "okay ich bin still"
	return "Ich sage doch garnichts"

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
			language = "de"
		logger.info("Verwende Sprache {}", language)
			
		logger.debug("Initialisiere Wake Word Erkennung...")
		self.wake_words = self.cfg['assistant']['wakewords']
		if not self.wake_words:
			self.wake_words = ['bumblebee']
		logger.debug("Wake Words sind {}", ','.join(self.wake_words))
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
		self.tts.say("Initialisierung abgeschlossen")
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
		
		# Initialisiere den Chatbot
		logger.info("Initialisiere Chatbot...")
		dialog_template_path = './dialogs/dialogs.template'
		if os.path.isfile(dialog_template_path):
			self.chat = Chat(dialog_template_path)
		else:
			logger.error('Dialogdatei konnte nicht in {} gefunden werden.', dialog_template_path)
			sys.exit(1)
		logger.info('Chatbot aus {} initialisiert.', dialog_template_path)
	
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
		try:
			while True:
			
				pcm = self.audio_stream.read(self.porcupine.frame_length)
				pcm_unpacked = struct.unpack_from("h" * self.porcupine.frame_length, pcm)		
				keyword_index = self.porcupine.process(pcm_unpacked)
				if keyword_index >= 0:
					logger.info("Wake Word {} wurde verstanden.", self.wake_words[keyword_index])
					self.is_listening = True
					
				# Spracherkennung
				if self.is_listening:
					if self.rec.AcceptWaveform(pcm):
						recResult = json.loads(self.rec.Result())
						
						speaker = self.__detectSpeaker__(recResult['spk'])
						if (speaker == None) and (self.allow_only_known_speakers == True):
							logger.info("Ich kenne deine Stimme nicht und darf damit keine Befehle von dir entgegen nehmen.")
							self.current_speaker = None
						else:
							if speaker:
								logger.debug("Sprecher ist {}", speaker)
							self.current_speaker = speaker
							self.current_speaker_fingerprint = recResult['spk']
							sentence = recResult['text']
							logger.debug('Ich habe verstanden "{}"', sentence)
							
							
							# Lasse den Assistenten auf die Spracheingabe reagieren.
							# Problem: Es wird das Default Template aufgerufen, wenn kein Intent erkannt wurde. Und das ist in Englisch.
							output = self.chat.respond(sentence)
							self.tts.say(output)
							
							self.is_listening = False
							self.current_speaker = None
					
		except KeyboardInterrupt:
			logger.debug("Per Keyboard beendet")
		finally:
			logger.debug('Beginne Aufräumarbeiten...')
			if self.porcupine:
				self.porcupine.delete()
				
			if self.audio_stream is not None:
				self.audio_stream.close()
				
			if self.pa is not None:
				self.pa.terminate()

if __name__ == '__main__':
	multiprocessing.set_start_method('spawn')

	va = VoiceAssistant()
	logger.info("Anwendung wurde gestartet")
	va.run()