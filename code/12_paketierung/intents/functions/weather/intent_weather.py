from loguru import logger
from chatbot import register_call
import global_variables
import random
import os
import yaml

import pyowm
from pyowm.utils.config import get_default_config
import geocoder

@register_call("weather")
def weather(session_id = "general", location=""):

	config_path = os.path.join('intents','functions','weather','config_weather.yml')
	cfg = None
	
	with open(config_path, "r", encoding='utf8') as ymlfile:
		cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
	
	if not cfg:
		logger.error("Konnte Konfigurationsdatei für das Wetter nicht lesen.")
		return ""

	LANGUAGE = global_variables.voice_assistant.cfg['assistant']['language']
	
	WEATHER_IS = random.choice(cfg['intent']['weather'][LANGUAGE]['weatheris'])
	HERE = cfg['intent']['weather'][LANGUAGE]['here']
	LOCATION_NOT_FOUND = cfg['intent']['weather'][LANGUAGE]['location_not_found']
	API_KEY = cfg['intent']['weather']['owm_api_key']
	
	# Konfiguration for die Wetter API
	config_dict = get_default_config()
	config_dict['language'] = LANGUAGE
		
	owm = pyowm.OWM(API_KEY, config_dict)
	weather_mgr = owm.weather_manager()
	
	location = location.strip()
	if (location == HERE) or (location == ""):
		g = geocoder.ip('me')
		w = weather_mgr.weather_at_coords(g.latlng[0], g.latlng[1]).weather
		return WEATHER_IS.format(g.city, w.detailed_status, str(w.temperature('celsius')['temp']))
	else:
		obs_list = weather_mgr.weather_at_places(location, 'like', limit=5)
		if len(obs_list) > 0:
			w = obs_list[0].weather
			return WEATHER_IS.format(location, w.detailed_status, str(w.temperature('celsius')['temp']))
	
	return LOCATION_NOT_FOUND.format(location)