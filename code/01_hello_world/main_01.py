from loguru import logger
import sys

# Gib alle Meldungen ab Level DEBUG aus. So können Level pro Modul gesteuert werden.
# Wir werden das in Zu
logger.remove()
logger.add(sys.stdout, level="DEBUG")

# Andere Level sind: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL

class VoiceAssistant():

	# Initialisiere die Instanz der Klasse VoiceAssistant
	# Methoden, deren Namen mit __ beginnen und enden weisen andere Entwickler darauf hin,
	# dass diese Methoden nur intern (innerhalb dieser Klasse/dieses Moduls verwendet werden sollen)
	def __init__(self):		
		logger.debug("VoiceAssistant wird initialisiert.")
	
	# Starte den VoiceAssistant
	def run(self):
		logger.info("VoiceAssistant Instanz wurde gestartet.")

# Wird dieses Script als python 01_main.py aufgerufen, ist die If-Bedingung erfüllt, und der Code dahinter
# wird ausgeführt.
if __name__ == '__main__':
	logger.info("VoiceAssistant wurde gestartet als {}", __name__)
	
	# Anlegen und Initiieren einer neuen Instanz des VoiceAssistant
	va = VoiceAssistant()
	
	logger.error("Sprachassistent kann noch nicht viel.")
	
	# Ausführen der Methode run()
	va.run()