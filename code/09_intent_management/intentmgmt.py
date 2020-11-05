import pip
import importlib
import importlib.util
import glob
import os
import sys
from pathlib import Path
import json
from snips_nlu import SnipsNLUEngine
from snips_nlu.default_configs import CONFIG_DE

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
		self.chatbotai_folders = [os.path.abspath(name) for name in glob.glob("./intents/chatbotai/*/")]
		self.functions_folders = [os.path.abspath(name) for name in glob.glob("./intents/functions/*/")]
		
		# Registriere Funktionen, die von snips-nlu und chatbotai aufgerufen werden
		function_count = 0
		for ff in self.functions_folders:
			print("Suche nach Funktionen in " + ff)
			req_file = os.path.join(ff, 'requirements.txt')
			if os.path.exists(req_file):
				install_result = self.install_requirements(req_file)
				if install_result == 0:
					print("Abhängigkeiten für " + ff + " erfolgreich installiert oder bereits vorhanden.")
			
			# Finde Python-Dateien, die mit Intent beginnen
			intent_files = glob.glob(os.path.join(ff, 'intent_*.py'))
			for infi in intent_files:
				print("Lade Intent Datei " + infi)
				spec = importlib.util.spec_from_file_location(Path(ff).name, infi)
				new_module = importlib.util.module_from_spec(spec)
				print("Modul " + str(new_module) + " geladen.")
				function_count +=1
				
		# Trainiere snips-nlu
		self.snips_nlu_engine = SnipsNLUEngine(Config=CONFIG_DE)
		#json_content = ''
		snips_nlu_count = 0
		snips_files = glob.glob(os.path.join("./intents/snips-nlu", '*.json'))
		for sf in snips_files:
			print("Verarbeite snips nlu training datei " + sf)
			with open(sf, 'r') as json_file:
				json_content = json.load(json_file)
				self.snips_nlu_engine = self.snips_nlu_engine.fit(json_content)
				print(self.snips_nlu_engine.dataset_metadata)
				snips_nlu_count +=1

		print(str(snips_nlu_count) + " Snips NLU files gefunden.")

		
		print("Snips NLU Training abgeschlossen")
					
				
	
	def evaluate(input: str, output: str):
		pass

		# Evaluiere ChatbotAI und snips-nlu
		
		# Wessen Confidence Score ist höher?
		
		# Rufe den Intent auf
		
		# Gib Ausgabe zurück