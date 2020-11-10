from datetime import datetime
import pytz

def gettime(country="deutschland"):

	country_timezone_map = {
		"deutschland": pytz.timezone('Europe/Berlin'),
		"england": pytz.timezone('Europe/London'),
		"frankreich": pytz.timezone('Europe/Paris'),
		"amerika": pytz.timezone('America/New_York'), # Hier gibt es natürlich mehre Zeitzonen...
		"china": pytz.timezone('Asia/Shanghai') # ...gilt traditionell auch hier, obwohl China eine öffentliche offizielle Zeitzone hat
	}

	now = datetime.now()
	timezone = country_timezone_map.get(country.lower())
	if timezone:
		now = datetime.now(timezone)
		return "Es ist " + str(now.hour) + " Uhr und " + str(now.minute) + " Minuten in " + country.capitalize() + "."
	return "Es ist " + str(now.hour) + " Uhr und " + str(now.minute) + " Minuten."
	
def asdasd():
	pass