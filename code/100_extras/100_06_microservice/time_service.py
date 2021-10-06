from flask import Flask, jsonify
from datetime import datetime
import pytz

app = Flask(__name__)

country_timezone_map = {
		'Europe/Berlin': ["deutschland", "germany"],
		'Europe/London': ["england", "großbrittanien", "great britain"],
		'Europe/Paris': ["frankreich", "france"],
		'America/New_York': ["amerika", "america"],
		'Asia/Shanghai': ["china"]
}
	
def gettimezone(place):
	timezone = None
	now = datetime.now()
	for c in country_timezone_map:
		if place.strip().lower() in country_timezone_map[c]:
			timezone = pytz.timezone(c)
			break
	return timezone

@app.route('/gettime', defaults={'place': 'default'})
@app.route('/gettime/<place>')
def gettime(place):
	timezone = gettimezone(place)
	output = ""
	if timezone:
		now = datetime.now(timezone)
		output = "Es ist " + str(now.hour) + " Uhr und " + str(now.minute) + " Minuten in " + place
	else:
		now = datetime.now()
		output = "Hier ist es " + str(now.hour) + " Uhr " + str(now.minute)
	
	return jsonify(output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)