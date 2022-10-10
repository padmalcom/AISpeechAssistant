import sys
from cx_Freeze import setup, Executable
import os

includefiles = ['config.yml', 'empty.wav', 'idle.png', 'initializing.png', 'listening.png', 'speaking.png', 'users.json', 'va.ico', 'vosk-model-de-0.6/', 'vosk-model-spk-0.4/', 'intents/']

build_exe_options = {"packages": ["pyttsx3.drivers.sapi5", "pip", "idna", "geocoder", "text2numde", "fuzzywuzzy", "dateutil", "pyowm", "wikipedia", "pycountry", "pycrfsuite", "pip._vendor.distlib", "pip._internal.commands.install", "pykeepass", "pynput", "scipy"], "excludes": [], "include_files": includefiles}

base = "Win32GUI"
#base = None


bdist_msi_options = {"install_icon":"va.ico", "summary_data": {"author": "Jonas Freiknecht"}}

setup(  name = "Sprachassistent",
        version = "1.0",
        description = "Mein erster Sprachassistent",
        options = {
			"build_exe": build_exe_options, # 1. Schritt, dann ffmpeg und _soundfile_data kopieren
			#"bdist_msi": bdist_msi_options # 2. Schritt (Schritt 1 auskommentieren, Schritt 2 einkommentieren)
		},
        executables = [Executable("main_12.py", base=base)])
		

# Anwendung muss einmal ausgeführt werden, um alle pakete in das environment zu bekommen	
# errors: sndfile library not found. Create _soundfile_data in lib dir and copy libsndfile64bit.dll
# add pyttsx3.drivers.sapi5 to packages
# multiprocessing.freeze_support() in main
# hinzufügen aller externer dateien
###### pip._internal.commands.install to packages
# pip to packages
# idna to packages
# base = "Win32GUI" to remove console

# 	sys.stdout = open('x.out', 'a')
#	sys.stderr = open('x.err', 'a')

# ffmpeg binaries https://anaconda.org/conda-forge/ffmpeg/4.3.1/download/win-64/ffmpeg-4.3.1-ha925a31_0.tar.bz2