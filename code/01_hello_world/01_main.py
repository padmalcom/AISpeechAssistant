import logging

logger = logging.getLogger(__name__)

# Gib alle Meldungen ab Level DEBUG (also DEBUG, aus
logging.basicConfig(level=logging.DEBUG)

# Benutze DEBUG level für den Logger dieses Moduls/dieser Datei
# NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
print("Gib nur die Meldungen dieses Moduls aus, die > DEBUG sind.")
logger.setLevel(logging.DEBUG)

class VoiceAssistant():

	# Initialisiere die Instanz der Klasse VoiceAssistant
	# Methoden, deren Namen mit __ beginnen und enden weisen andere Entwickler darauf hin,
	# dass diese Methoden nur intern (innerhalb dieser Klasse/dieses Moduls verwendet werden sollen)
	def __init__(self):		
		logger.info("VoiceAssistant wird initialisiert.")
	
	# Starte den VoiceAssistant
	def run(self):
		logger.info("VoiceAssistant Instanz wurde gestartet.")

# Wird dieses Script als python 01_main.py aufgerufen, ist die If-Bedingung erfüllt, und der Code dahinter
# wird ausgeführt.
if __name__ == '__main__':
	logger.info("VoiceAssistant wurde gestartet als %s", __name__)
	
	# Anlegen und Initiieren einer neuen Instanz des VoiceAssistant
	va = VoiceAssistant()
	
	# Ausführen der Methode run()
	va.run()