from loguru import logger
import pyttsx3
import logging

logger = logging.getLogger(__name__)


# Unterdrücke Logausgaben der Hintergrunddienste von pyttsx3
logging.getLogger('comtypes._comobject').setLevel(logging.WARNING)

class VoiceAssistant():

	def __init__(self):
		logger.info("Initialisiere VoiceAssistant...")
		
		logger.info("Initialisiere Sprachausgabe...")
		self.tts = pyttsx3.init();
		
		# Ausgabe aller Sprachengines und Sprachpakete
		voices = self.tts.getProperty('voices')
		for voice in voices:
			logger.info(voice)
			
		# Wählt bitte eine der ausgegebenen IDs und setzt die voiceId entsprechend
		voiceId = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_DE-DE_HEDDA_11.0"
		self.tts.setProperty('voice', voiceId)
		self.tts.say("Initialisierung abgeschlossen");
		self.tts.runAndWait();
		logger.debug("Sprachausgabeninitialisierung abgeschlossen.")
		
		
	def run(self):
		logger.info("VoiceAssistant Instanz wurde gestartet.")
		self.tts.say("Ich bin bereit.");
		self.tts.runAndWait();


if __name__ == '__main__':
	logger.info("Anwendung wurde gestartet")
	va = VoiceAssistant()
	va.run()