from loguru import logger
import yaml
import sys

from TTS import Voice
import multiprocessing

CONFIG_FILE = "config.yml"

class VoiceAssistant():

	def __init__(self):
		logger.info("Initialisiere VoiceAssistant...")
		
		# Lese Konfigurationsdatei
		logger.debug("Lese Konfiguration...")
		
		# Verweise lokal auf den globalen Kontext und hole die Variable CONFIG_FILE
		global CONFIG_FILE
		with open(CONFIG_FILE, "r", encoding='utf8') as ymlfile:
			# Lade die Konfiguration im YAML-Format
			self.cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
		if self.cfg:
			logger.debug("Konfiguration gelesen.")
		else:
			# Konnto keine Konfiguration gefunden werden? Dann beende die Anwendung
			logger.debug("Konfiguration konnte nicht gelesen werden.")
			sys.exit(1)
		language = self.cfg['assistant']['language']
		if not language:
			language = "German"
		logger.info("Verwende Sprache {}", language)
		
		# Initialisiere TTS
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
		
	
	def run(self):
		logger.info("VoiceAssistant Instanz wurde gestartet.")

if __name__ == '__main__':
	# Für Windows nutzen wir spawn, nicht fork
	# Siehe: https://docs.python.org/3/library/multiprocessing.html
	multiprocessing.set_start_method('spawn')

	va = VoiceAssistant()
	logger.info("Anwendung wurde gestartet")
	va.run()