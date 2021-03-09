# AISpeechAssistant

## TODOs

### Done
- name every intent and add authentication script

### Open
- Englische Templates und deren Verwaltung
- 12 paketierung (conda constructor) pyflakes um zu checken was gebraucht wird und was nicht
- Sprechblase in 12, um Dialoge zu visualisieren
- Schwarm Cluster - jeder bekommt einen Namen und ein Response-Ton
- Passwort Skill: "Passwort PKI"
- Geburtstagsskill

## Chapters
1. Introduction
	1.1 Speech Assistant to support Life and Home Automation
	1.2 Challenges:
		- To create an offline bot that does not collect data
		- To create a multi language bot
		- To create an intelligent bot that interprets commands correctly
		- Allow for extensibility
	1.3 Technology
		- Various possibilities of platforms to use for such a bot (Raspberry Pi, PC, Mobile Phone)
		- Python because it is the language which appears most in AI research
		- Portability: What if you want to carry your bot around?
2. Setup the environment
	2.1 What we need
	2.2 Installing Anaconda
	2.3 Installing an IDE
		- We use Notepad++ what might be uncommon but is a good way to learn (the hard way)
	2.4 Structure of this course
		- One environement for each chapter
		- Sometimes we change technologies to learn
3. Configuration files
	3.1 Why configuration files?
		- Simple parametrization
		- Languages
		- Right management
	3.2 Creating a config file
4. Voice
	4.1 Why we start with voice? It is fun, it is one of the trickiest parts
	4.2 Text-to-Speech
		4.2.1 Using pyttsx
		4.2.2 Different speech APIs
		4.2.2 Languages
		4.2.3 Make it stop!
		4.2.4 (Optional) Training your own voice
	4.3 Wake word detection
		4.3.1 Using pre-defined words using Porcupine
		4.3.2 Training an own keyword
		4.3.3 (Optional) Activation by key
	4.4 Speech-To-Text
		4.4.1 Implementing VOSK
		4.4.2 Languages
		4.4.2 Combining PyAudio, Porcupine and VOSK
	4.5 Speaker detection
5. Intents
	5.1 What we will do
		- Some online, some offline functions
		- Weather forecast?
		- What is/ who is?
		- Calendar/ Reminder
		- What sound does a X make?
		- Turn on the lights
		- Where am I?
		- Sending mails
		- Chatting
		- Tell a story
		- My next appointments
		
	5.1 Implementing a static intent recognizer
	5.2 Training an ML intent recognizer
	5.3 Allow to extend intents
6. User Management
	6.1 Storing speaker data
	6.2 Allow only known speakers
	6.3 Allow to add speakers
	6.4 Guessing a speakers gender
	6.5 Remembering speakers and last interaction time
	6.6 Birthdays
		- Ask for birthday
		- Remind others to birthdays
5. Computer Vision
	5.1 Identify speaker
6. Messaging
	6.1 Implement a chat with riot
	