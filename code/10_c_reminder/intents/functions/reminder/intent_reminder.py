from datetime import datetime
import pytz
import global_variables
import os
import random
import yaml

from dateutil.parser import parse
from num2words import num2words

def spoken_date(datetime, lang):
	hours = str(datetime.hour)
	minutes = "" if datetime.minute == 0 else str(datetime.minute)
	day = num2words(datetime.day, lang=lang, to="ordinal")
	month = num2words(datetime.month, lang=lang, to="ordinal")
	year = "" if datetime.year == datetime.now().year else str(datetime.year)
	
	# Anpassung an den deutschen Casus
	if lang == 'de':
		day += 'n'
		month += 'n'
		
	return hours, minutes, day, month, year

def reminder(time=None, reminder_to=None, reminder_infinitive=None):

	# Hole den aktuellen Sprecher, falls eine persönliche Erinnerung stattfinden soll
	speaker = global_variables.voice_assistant.current_speaker
	
	config_path = os.path.join('intents','functions','reminder','config_reminder.yml')
	cfg = None
	with open(config_path, "r", encoding='utf8') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
	
	if not cfg:
		logger.error("Konnte Konfigurationsdatei für reminder nicht lesen.")
		return ""
		
	# Holen der Sprache aus der globalen Konfigurationsdatei
	LANGUAGE = global_variables.voice_assistant.cfg['assistant']['language']
	NO_TIME_GIVEN = random.choice(cfg['intent']['reminder'][LANGUAGE]['no_time_given'])

	print(time)
	print(reminder_to)
	print(reminder_infinitive)
		
	# Bereite das Ergebnis vor
	result = ""
	if speaker:
		result = speaker + ', '

	# Wurde keine Uhrzeit angegeben?
	if not time:
		return result + NO_TIME_GIVEN

	# Wir machen uns das Parsing des Datums-/Zeitwertes leicht
	parsed_time = parse(time)
	
	# Generiere das gesprochene Datum
	hours, minutes, day, month, year = spoken_date(parsed_time, LANGUAGE)
	
	print(hours)
	print(minutes)
	print(day)
	print(month)
	print(year)
	

	# Am selben Tag wie heute?
	if datetime.now().date() == parsed_time.date():
		if reminder_to:
			print(1)
			SAME_DAY_TO = random.choice(cfg['intent']['reminder'][LANGUAGE]['reminder_same_day_to'])
			SAME_DAY_TO = SAME_DAY_TO.format(hours, minutes, reminder_to)
			result = result + " " + SAME_DAY_TO
		elif reminder_infinitive:
			print(2)
			SAME_DAY_INFINITIVE = random.choice(cfg['intent']['reminder'][LANGUAGE]['reminder_same_day_infinitive'])
			SAME_DAY_INFINITIVE = SAME_DAY_INFINITIVE.format(hours, minutes, reminder_infinitive)
			result = result + " " + SAME_DAY_INFINITIVE
		else:
			print(3)
			# Es wurde nicht angegeben, an was erinnert werden soll
			SAME_DAY_NO_ACTION = random.choice(cfg['intent']['reminder'][LANGUAGE]['reminder_same_day_no_action'])
			SAME_DAY_NO_ACTION = SAME_DAY_NO_ACTION.format(hours, minutes)
			result = result + " " + SAME_DAY_NO_ACTION
	else:
		if reminder_to:
			print(4)
			TO = random.choice(cfg['intent']['reminder'][LANGUAGE]['reminder_to'])
			TO = TO.format(day, month, year, hours, minutes, reminder_to)
			result = result + " " + TO
		elif reminder_infinitive:
			print(5)
			INFINITIVE = random.choice(cfg['intent']['reminder'][LANGUAGE]['reminder_infinitive'])
			INFINITIVE = INFINITIVE.format(day, month, year, hours, minutes, reminder_infinitive)
			result = result + " " + INFINITIVE
		else:
			Print(6)
			# Es wurde nicht angegeben, an was erinnert werden soll
			NO_ACTION = random.choice(cfg['intent']['reminder'][LANGUAGE]['reminder_no_action'])
			NO_ACTION = NO_ACTION.format(day, month, year, hours, minutes)
			result = result + " " + NO_ACTION
			
	return result