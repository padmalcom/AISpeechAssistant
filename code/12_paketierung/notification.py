from notifypy import Notify
import constants

class Notification:

	def show(title, message, ttl):
		notification = Notify()
		notification.title = title
		notification.message = message
		notification.audio = constants.find_data_file("empty.wav")
		notification.icon = constants.find_data_file("va.ico")
		notification.application_name = "Sprachasssistent"
		notification.send()