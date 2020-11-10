from loguru import logger
import pip
import importlib
import importlib.util
import glob
import os
import sys
from pathlib import Path
from snips_nlu import SnipsNLUEngine
from snips_nlu.default_configs import CONFIG_DE
from snips_nlu.dataset import Dataset
from chatbot import Chat, register_call
import json

intentMgmt = None

@register_call("default_snips_nlu_handler")
def default_snips_nlu_handler(session, text):
	parsing = intentMgmt.nlu_engine.parse(text)
	print(parsing)
	output = "Ich verstehe deine Frage nicht. Kannst du sie umformulieren?"
	
	# Schaue, ob es einen Intent gibt, der zu dem NLU intent passt
	intent_found = False
	output = "a"
	for intent in intentMgmt.dynamic_intents:
		
		# Wenn wir hier nicht auf die Wahrscheinlichkeit prüfen, trifft dieser Intent immer ein, egal wie
		# wahrscheinlich er ist.
		if (parsing["intent"]["intentName"].lower() == intent.lower()) and (parsing["intent"]["probability"] > 0.7):
			intent_found = True
			
			# Parse alle Parameter
			arguments = dict()
			for slot in parsing["slots"]:
				arguments[slot["entity"]] = slot["rawValue"]
				
			# Rufe Methode dynamich mit der Parameterliste auf
			argument_string = json.dumps(arguments)
			logger.debug("Rufe {} auf mit den Argumenten {}.", intent, argument_string)
			output = getattr(globals()[intent], intent)(**arguments)
			
			# Suche nicht weiter
			break
	
	return output
		
class IntentMgmt:

	def install(self, package):
		if hasattr(pip, 'main'):
			pip.main(['install', package])
		else:
			pip._internal.main(['install', package])
			
	def install_requirements(self, filename):
		retcode = 0
		with open(filename, 'r') as f:
			for line in f:
				pipcode = pip.main(['install', line.strip()])
				retcode = retcode or pipcode
		return retcode
		
	def get_count(self):
		return 1

	def __init__(self):
		self.functions_folders = [os.path.abspath(name) for name in glob.glob("./intents/functions/*/")]
		self.dynamic_intents = []
		# Registriere Funktionen, die von snips-nlu und chatbotai aufgerufen werden
		function_count = 0
		for ff in self.functions_folders:
			logger.debug("Suche nach Funktionen in {}...", ff)
			req_file = os.path.join(ff, 'requirements.txt')
			if os.path.exists(req_file):
				install_result = self.install_requirements(req_file)
				if install_result == 0:
					logger.debug("Abhängigkeiten für {} erfolgreich installiert oder bereits vorhanden.", ff)
			
			# Finde Python-Dateien, die mit Intent beginnen
			intent_files = glob.glob(os.path.join(ff, 'intent_*.py'))
			for infi in intent_files:
				logger.debug("Lade Intent-Datei {}...", infi)
								
				name = infi.strip('.py')
				name = "intents.functions." + Path(ff).name + ".intent_" + Path(ff).name
				name = name.replace(os.path.sep, ".")
				
				logger.debug("Importiere modul {}...", name)
				globals()[Path(ff).name] = importlib.import_module(name)
				logger.debug("Modul {} geladen.", str(Path(ff).name))
				self.dynamic_intents.append(str(Path(ff).name))
				function_count +=1
				
		# Trainiere snips-nlu
		self.snips_nlu_engine = SnipsNLUEngine(Config=CONFIG_DE)
		snips_files = glob.glob(os.path.join("./intents/snips-nlu", '*.yaml'))
		dataset = Dataset.from_yaml_files("de", snips_files)
		nlu_engine = SnipsNLUEngine(config=CONFIG_DE)
		self.nlu_engine = nlu_engine.fit(dataset)
		logger.info("{} Snips NLU files gefunden.", len(snips_files))
		if not self.nlu_engine:
			logger.error("Konnte Dialog-Engine nicht laden.")
		else:
			logger.debug("Dialog Metadaten: {}.", self.nlu_engine.dataset_metadata)
	
		logger.debug("Snips NLU Training abgeschlossen")
		
		# chatbotai
		logger.info("Initialisiere Chatbot...")
		dialog_template_path = './intents/chatbotai/dialogs.template'
		if os.path.isfile(dialog_template_path):
			# Wir müssen hier kein Default template setzen, da dieses durch den Wildcard Intent nie aufgerufen wird.
			# Vorher wird snips nlu aufgerufen.
			self.chat = Chat(dialog_template_path)
		else:
			logger.error('Dialogdatei konnte nicht in {} gefunden werden.', dialog_template_path)
		logger.info('Chatbot aus {} initialisiert.', dialog_template_path)
		
		global intentMgmt
		intentMgmt = self
				
	
	def process(self, text, speaker):
	
		# Evaluiere ChatbotAI, wenn keines der strikten Intents greift, wird die Eingabe über die dialogs.template
		# automatisch an snips nlu umgeleitet.
		return self.chat.respond(text)