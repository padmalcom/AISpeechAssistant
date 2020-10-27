import logging
import pyttsx3
from TTS import Voice
import multiprocessing

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)
logging.getLogger('comtypes._comobject').setLevel(logging.WARNING)

class VoiceAssistant():

	def __init__(self):
		logger.info("Initialisiere VoiceAssistant...")
		
		logger.info("Initialisiere Sprachausgabe...")
		self.tts = Voice()
		voices = self.tts.get_voice_keys_by_language("German")
		if len(voices) > 0:
			logger.info('Stimme %s gesetzt.', voices[0])
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
	try:
		logger.info("Anwendung wurde gestartet")
		va.run()