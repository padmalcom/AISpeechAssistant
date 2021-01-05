pyinstaller ^
	--noconfirm ^
	--additional-hooks-dir "hooks" ^
	--name AISpeechAssistant ^
	--add-data="dylib;dylib" ^
	--add-data="pvporcupine;pvporcupine" ^
	--add-data="idle.png;." ^
	--add-data="initializing.png;." ^
	--add-data="listening.png;." ^
	--add-data="speaking.png;." ^
	--add-data="config.yml;." ^
	--add-data="users.json;." ^
	--add-data="intents;intents" ^
	--add-data="vosk-model-de-0.6;vosk-model-de-0.6" ^
	--add-data="vosk-model-spk-0.4;vosk-model-spk-0.4" ^
	--icon "logo.ico" ^
	--hidden-import=pyttsx3.drivers ^
	--hidden-import=pyttsx3.drivers.sapi5 ^
	--hidden-import=pip._internal.commands.install ^
	--hidden-import=pytz ^
	--hidden-import=geocoder ^
	--hidden-import=fuzzywuzzy.fuzz ^
	--hidden-import=text2numde ^
	--hidden-import=sounddevice ^
	--hidden-import=dateutil ^
	--hidden-import=dateutil.parser ^
	--hidden-import=num2words ^
	--hidden-import=text2numde ^
	--hidden-import=pyowm ^
	--hidden-import=pycountry ^
	--hidden-import=wikipedia ^
	--log-level DEBUG ^
	main_12.py

REM	--windowed ^	