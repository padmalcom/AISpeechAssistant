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
	--icon "logo.ico" ^
	main_12.py

REM	--windowed ^	