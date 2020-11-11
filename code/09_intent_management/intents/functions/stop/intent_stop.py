from chatbot import register_call
import global_variables

# Spezieller Intent, der Zugriff auf va braucht	
@register_call("stop")
def stop(dummy=0, session_id = "general"):

	if global_variables.va.tts.is_busy():
		global_variables.tts.stop()
		return "okay ich bin still"
	return "Ich sage doch garnichts"