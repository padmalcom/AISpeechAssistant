from loguru import logger
import yaml
import time
import pvporcupine
import pyaudio
import struct
import os
import sys
from vosk import Model, SpkModel, KaldiRecognizer
import json
import numpy as np
from usermgmt import UserMgmt
import io

import global_variables
from TTS import Voice
import multiprocessing
from intentmgmt import IntentMgmt
from audioplayer import AudioPlayer

# UI Komponenten für das Tray Icon
import wx.adv
import wx

# Konstanten für das Tray Icon
import constants

from notification import Notification

CONFIG_FILE = "config.yml"

# Eine Klasse, die die Logik unseres TrayIcons abbildet.
class TaskBarIcon(wx.adv.TaskBarIcon):
	def __init__(self, frame):
		self.frame = frame
		super(TaskBarIcon, self).__init__()
		self.set_icon(constants.TRAY_ICON_INITIALIZING, constants.TRAY_TOOLTIP + ": Initialisiere...")
		
	# Methode, um Menü-Einträge hinzuzufügen.
	def create_menu_item(self, menu, label, func):
		item = wx.MenuItem(menu, -1, label)
		menu.Bind(wx.EVT_MENU, func, id=item.GetId())
		menu.Append(item)
		return item

	# Wir erstellen ein Menü-Eintrag, der bei einem Rechtsklick gezeigt wird.
	def CreatePopupMenu(self):
		menu = wx.Menu()
		self.create_menu_item(menu, 'Beenden', self.on_exit)
		return menu

	# Ändern des Icons und des Hilfetextes
	def set_icon(self, path, tooltip=constants.TRAY_TOOLTIP):
		icon = wx.Icon(path)
		self.SetIcon(icon, tooltip)
	
	# Beenden der Applikation über das Menü
	def on_exit(self, event):
		if global_variables.voice_assistant:
			global_variables.voice_assistant.terminate()
			wx.CallAfter(self.Destroy)
			self.frame.Close()

# Die Klasse enthält die Logik für den Part der UI			
class MainApp(wx.App):
	def OnInit(self):
		frame = wx.Frame(None)
		self.SetTopWindow(frame)
		self.icon = TaskBarIcon(frame)
		self.Bind(wx.EVT_CLOSE, self.onCloseWindow)
		
		# Erstelle einen Timer, der die Hauptschleife unseres Assistenten ausführt
		self.timer = wx.Timer(self)
		self.Bind(wx.EVT_TIMER, self.update, self.timer)
		
		return True
		
	def update(self, event):
		if global_variables.voice_assistant:
			global_variables.voice_assistant.loop()

	def onCloseWindow(self, evt):
		self.icon.Destroy()
		evt.Skip()	

class VoiceAssistant:

	def __init__(self):
		logger.info("Initialisiere VoiceAssistant...")
		
		logger.info("Initialisiere UI...")
		# Wir kontrollieren die App über das TrayIcon, deswegen löschen
		# wir alle Signale zum Beenden der Anwendung per STRG+C.
		# Weiterhin leiten wir die Ausgabe der Konsole und die dazugehörigen
		# Fehlermeldungen in die Datei "log.txt" um.
		self.app = MainApp(clearSigInt=False, redirect=True, filename='log.txt')
				
		logger.debug("Lese Konfiguration...")
		
		global CONFIG_FILE
		with open(CONFIG_FILE, "r", encoding='utf8') as ymlfile:
			self.cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
		if self.cfg:
			logger.debug("Konfiguration gelesen.")
		else:
			logger.debug("Konfiguration konnte nicht gelesen werden.")
			sys.exit(1)
		language = self.cfg['assistant']['language']
		if not language:
			language = "de"
		logger.info("Verwende Sprache {}", language)
		
		self.show_balloon = self.cfg['assistant']['show_balloon']
			
		logger.debug("Initialisiere Wake Word Erkennung...")
		self.wake_words = self.cfg['assistant']['wakewords']
		if not self.wake_words:
			self.wake_words = ['bumblebee']
		logger.debug("Wake words are {}", ','.join(self.wake_words))
		self.porcupine = pvporcupine.create(keywords=self.wake_words)
		logger.debug("Wake Word Erkennung wurde initialisiert.")
		
		logger.debug("Initialisiere Audioeingabe...")
		self.pa = pyaudio.PyAudio()
		
		self.audio_stream = self.pa.open(
			rate=self.porcupine.sample_rate,
			channels=1,
			format=pyaudio.paInt16,
			input=True,
			frames_per_buffer=self.porcupine.frame_length,
			input_device_index=0)
		logger.debug("Audiostream geöffnet.")
		
		# Lese Lautstärke
		self.volume = self.cfg["assistant"]["volume"]
		self.silenced_volume = self.cfg["assistant"]["silenced_volume"]

		logger.info("Initialisiere Sprachausgabe...")
		self.tts = Voice()
		voices = self.tts.get_voice_keys_by_language(language)
		if len(voices) > 0:
			logger.info('Stimme {} gesetzt.', voices[0])
			self.tts.set_voice(voices[0])
		else:
			logger.warning("Es wurden keine Stimmen gefunden.")
		self.tts.set_volume(self.volume)
		self.tts.say("Sprachausgabe aktiviert.")
		if self.show_balloon:
			Notification.show('Initialisierung', 'Sprachausgabe aktiviert', ttl=4000)
			
		logger.debug("Sprachausgabe initialisiert")
		
		logger.info("Initialisiere Spracherkennung...")
		stt_model = Model('./vosk-model-de-0.6')
		speaker_model = SpkModel('./vosk-model-spk-0.4')
		self.rec = KaldiRecognizer(stt_model, speaker_model, 16000)
		self.is_listening = False
		logger.info("Initialisierung der Spracherkennung abgeschlossen.")
		
		logger.info("Initialisiere Benutzerverwaltung...")
		self.user_management = UserMgmt(init_dummies=True)
		self.allow_only_known_speakers = self.cfg["assistant"]["allow_only_known_speakers"]
		logger.info("Benutzerverwaltung initialisiert")
		
		# Initialisiere den Audio-Player
		self.audio_player = AudioPlayer()
		self.audio_player.set_volume(self.volume)
				
		logger.info("Initialisiere Intent-Management...")
		self.intent_management = IntentMgmt()
		logger.info('{} intents geladen', self.intent_management.get_count())
		
		# Erzeuge eine Liste, die die Callback Funktionen vorhält
		self.callbacks = self.intent_management.register_callbacks()
		logger.info('{} callbacks gefunden', len(self.callbacks))
		self.tts.say("Initialisierung abgeschlossen")
		if self.show_balloon:
			Notification.show('Initialisierung', 'Abgeschlossen', ttl=4000)

		self.app.icon.set_icon(constants.TRAY_ICON_IDLE, constants.TRAY_TOOLTIP + ": Bereit")
		timer_start_result = self.app.timer.Start(milliseconds=1, oneShot=wx.TIMER_CONTINUOUS)
		logger.info("Timer erfolgreich gestartet? {}", timer_start_result)
	
	# Finde den besten Sprecher aus der Liste aller bekannter Sprecher aus dem User Management
	def __detectSpeaker__(self, input):
		bestSpeaker = None
		bestCosDist = 100
		for speaker in self.user_management.speaker_table.all():
			nx = np.array(speaker.get('voice'))
			ny = np.array(input)
			cosDist = 1 - np.dot(nx, ny) / np.linalg.norm(nx) / np.linalg.norm(ny)
			if (cosDist < bestCosDist):
				if (cosDist < 0.3):
					bestCosDist = cosDist
					bestSpeaker = speaker.get('name')
		return bestSpeaker
	
	def terminate(self):
		logger.debug('Beginne Aufräumarbeiten...')
		
		# Stoppe den Timer
		self.app.timer.Stop()
		
		# Speichern der Konfiguration
		global_variables.voice_assistant.cfg["assistant"]["volume"] = global_variables.voice_assistant.volume
		global_variables.voice_assistant.cfg["assistant"]["silenced_volume"] = global_variables.voice_assistant.silenced_volume
		with open(CONFIG_FILE, 'w') as f:
			yaml.dump(global_variables.voice_assistant.cfg, f, default_flow_style=False, sort_keys=False)		
		
		if global_variables.voice_assistant.porcupine:
			global_variables.voice_assistant.porcupine.delete()
			
		if global_variables.voice_assistant.audio_stream is not None:
			global_variables.voice_assistant.audio_stream.close()
			
		if global_variables.voice_assistant.audio_player is not None:
			global_variables.voice_assistant.audio_player.stop()			
			
		if global_variables.voice_assistant.pa is not None:
			global_variables.voice_assistant.pa.terminate()			
	
	# Jegliche Logik aus der Main-Methode wird nun in die Update-Methode verlagert
	def loop(self):		
		pcm = global_variables.voice_assistant.audio_stream.read(global_variables.voice_assistant.porcupine.frame_length)
		pcm_unpacked = struct.unpack_from("h" * global_variables.voice_assistant.porcupine.frame_length, pcm)		
		keyword_index = global_variables.voice_assistant.porcupine.process(pcm_unpacked)
		if keyword_index >= 0:
			logger.info("Wake Word {} wurde verstanden.", global_variables.voice_assistant.wake_words[keyword_index])
			global_variables.voice_assistant.is_listening = True
			
		# Spracherkennung
		if global_variables.voice_assistant.is_listening:
			if not global_variables.voice_assistant.tts.is_busy():
				self.app.icon.set_icon(constants.TRAY_ICON_LISTENING, constants.TRAY_TOOLTIP + ": Ich höre...")
		
			# Spielt derzeit Musik oder sonstiges Audio? Dann setze die Lautstärke runter
			if global_variables.voice_assistant.audio_player.is_playing():
				global_variables.voice_assistant.audio_player.set_volume(global_variables.voice_assistant.silenced_volume)
					
			if global_variables.voice_assistant.rec.AcceptWaveform(pcm):
				recResult = json.loads(global_variables.voice_assistant.rec.Result())
				
				speaker = global_variables.voice_assistant.__detectSpeaker__(recResult['spk'])
				if (speaker == None) and (global_variables.voice_assistant.allow_only_known_speakers == True):
					logger.info("Ich kenne deine Stimme nicht und darf damit keine Befehle von dir entgegen nehmen.")
					global_variables.voice_assistant.current_speaker = None
				else:
					if speaker:
						logger.debug("Sprecher ist {}", speaker)
					global_variables.voice_assistant.current_speaker = speaker
					global_variables.voice_assistant.current_speaker_fingerprint = recResult['spk']
					sentence = recResult['text']
					logger.debug('Ich habe verstanden "{}"', sentence)
					
					# Lasse den Assistenten auf die Spracheingabe reagieren
					output = global_variables.voice_assistant.intent_management.process(sentence, speaker)
					global_variables.voice_assistant.tts.say(output)
					if self.show_balloon:
						Notification.show("Interaktion", "Eingabe (" + speaker + "): " + sentence + ". Ausgabe: " + output, ttl=4000)
					
					global_variables.voice_assistant.is_listening = False
					global_variables.voice_assistant.current_speaker = None
		
		# Wird derzeit nicht zugehört?
		else:
		
			# Reaktiviere is_listening, wenn der Skill weitere Eingaben erforder
			if not global_variables.context is None:
				# ... aber erst, wenn ausgeredet wurde
				if not global_variables.voice_assistant.tts.is_busy():
					global_variables.voice_assistant.is_listening = True
			else:		
				if not global_variables.voice_assistant.tts.is_busy():
					self.app.icon.set_icon(constants.TRAY_ICON_IDLE, constants.TRAY_TOOLTIP + ": Bereit")
				# Setze die Lautstärke auf Normalniveau zurück
				global_variables.voice_assistant.audio_player.set_volume(global_variables.voice_assistant.volume)
						
				# Prozessiere alle registrierten Callback Funktionen, die manche Intents
				# jede Iteration benötigen
				for cb in global_variables.voice_assistant.callbacks:
					output = cb()
					
					# Gibt die Callback Funktion einen Wert zurück? Dann versuche
					# ihn zu sprechen.
					if output:
						if not global_variables.voice_assistant.tts.is_busy():
					
							# Wird etwas abgespielt? Dann schalte die Lautstärke runter
							if global_variables.voice_assistant.audio_player.is_playing():
								global_variables.voice_assistant.audio_player.set_volume(global_variables.voice_assistant.audio_player.set_volume(global_variables.voice_assistant.silenced_volume))
							
							# 
							global_variables.voice_assistant.tts.say(output)
							if self.show_balloon:
								Notification.show('Callback', output, ttl=4000)
							
							# Wir rufen die selbe Funktion erneut auf und geben mit,
							# dass der zu behandelnde Eintrag abgearbeitet wurde.
							# Im Falle der Reminder-Funktion wird dann z.B. der Datenbankeintrag
							# für den Reminder gelöscht
							cb(True)
							
							# Zurücksetzen der Lautstärke auf Normalniveau
							global_variables.voice_assistant.audio_player.set_volume(global_variables.voice_assistant.volume)

if __name__ == '__main__':
	sys.stdout = open('x.out', 'a')
	sys.stderr = open('x.err', 'a')
	multiprocessing.freeze_support()
	multiprocessing.set_start_method('spawn')
	
	# Initialisiere den Voice Assistant
	global_variables.voice_assistant = VoiceAssistant()
	
	# Starten der Hauptschleife der UI, die auch unseren Timer beinhaltet.
	global_variables.voice_assistant.app.MainLoop()
