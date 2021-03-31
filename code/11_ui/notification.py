from notifypy import Notify

class Notification:

	def show(title, message, ttl):
		notification = Notify()
		notification.title = title
		notification.message = message
		notification.audio = "empty.wav"
		notification.icon = "va.ico"
		notification.application_name = "Sprachasssistent"
		notification.send()