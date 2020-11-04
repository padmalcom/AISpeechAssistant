import pip
import importlib
import glob
import os

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
		self.snips_nlu_folders = [os.path.abspath(name) for name in glob.glob("./intents/snips-nlu/*/")]
		self.chatbotai_folders = [os.path.abspath(name) for name in glob.glob("./intents/chatbotai/*/")]
		self.functions_folders = [os.path.abspath(name) for name in glob.glob("./intents/functions/*/")]
		
		# install functions
		for ff in self.functions_folders:
			print("Suche nach Funktionen in " + ff)
			req_file = os.path.join(ff, 'requirements.txt')
			if os.path.exists(req_file):
				install_result = self.install_requirements(req_file)
				print(install_result)
				if install_result:
					print("Abhängigkeiten für " + ff + " erfolgreich installiert.")
			
			# Finde Python-Dateien, die mit Intent beginnen
			intent_files = glob.glob(os.path.join(ff, 'intent_*.py'))
			for infi in intent_files:
				print("Lade Intent Datei " + infi)
				new_module = importlib.import_module(infi)
				print("Modul " + new_module + " geladen.")
				
	
	def evaluate(input: str, output: str):
		pass

		# Evaluiere ChatbotAI und snips-nlu
		
		# Wessen Confidence Score ist höher?
		
		# Rufe den Intent auf
		
		# Gib Ausgabe zurück