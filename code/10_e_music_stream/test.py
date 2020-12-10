from audioplayer import AudioPlayer
import time

if __name__ == '__main__':

	ap = AudioPlayer()
	try:
		ap.set_volume(1)
		ap.play_stream("http://st01.dlf.de/dlf/01/64/mp3/stream.mp3")
		#ap.play_file(r"C:\Users\jfrei\AISpeechAssistant\code\10_e_music_stream\intents\functions\animalsounds\animals\cat.ogg")
		time.sleep(5)
		ap.set_volume(0.1)
		ap.stop()
		ap.play_stream("http://st01.dlf.de/dlf/01/64/mp3/stream.mp3")
		#ap.play_file(r"C:\Users\jfrei\AISpeechAssistant\code\10_e_music_stream\intents\functions\animalsounds\animals\cat.ogg")
		time.sleep(4)
		ap.stop()
	finally:
		if ap:
			ap.stop()