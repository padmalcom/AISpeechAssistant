from loguru import logger
import yaml
import time
import pyaudio
import struct
import os
import sys

from vosk import Model, SpkModel, KaldiRecognizer
import json
import text2numde 

from TTS import Voice
import multiprocessing

CONFIG_FILE = "config.yml"

SAMPLE_RATE = 16000
FRAME_LENGTH = 512

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
		
		logger.debug("Initialisiere Audioeingabe...")
		self.pa = pyaudio.PyAudio()
		
		self.audio_stream = self.pa.open(
			rate=SAMPLE_RATE,
			channels=1,
			format=pyaudio.paInt16,
			input=True,
			frames_per_buffer=FRAME_LENGTH,
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
		
		# Initialisiere Spracherkennung
		logger.info("Initialisiere Spracherkennung...")
		stt_model = Model('./vosk-model-de-0.6')
		speaker_model = SpkModel('./vosk-model-spk-0.4')
		self.rec = KaldiRecognizer(stt_model, speaker_model, 16000)

		logger.info("Initialisierung der Spracherkennung abgeschlossen.")
			
	def run(self):
		logger.info("VoiceAssistant Instanz wurde gestartet.")
		try:
			while True:
			
				pcm = self.audio_stream.read(FRAME_LENGTH)
					
				if self.rec.AcceptWaveform(pcm):
					recResult = json.loads(self.rec.Result())
						
					# Hole das Resultat aus dem JSON Objekt
					sentence = recResult['text']
					logger.debug('Ich habe verstanden "{}"', sentence)
					
					if sentence.lower().startswith("kevin"):
						sentence = sentence [5:] # Schneide Kevin am Anfang des Satzes weg
						sentence = sentence.strip() # Entferne Leerzeichen am Anfang und Ende des Satzes
						logger.info("Prozessiere Befehl {}.", sentence)
					
		except KeyboardInterrupt:
			logger.debug("Per Keyboard beendet")
		finally:
			logger.debug('Beginne Aufräumarbeiten...')
				
			if self.audio_stream is not None:
				self.audio_stream.close()
				
			if self.pa is not None:
				self.pa.terminate()

if __name__ == '__main__':
	multiprocessing.set_start_method('spawn')

	va = VoiceAssistant()
	logger.info("Anwendung wurde gestartet")
	va.run()				