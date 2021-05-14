# AISpeechAssistant

English: This repository contains the code for the tutorial "creating an ai speech assistant" and works without the connection to any cloud services.

Deutsch: Dieses Repository ist die Code-Basis für den dazugehörigen Kurs zur Entwicklung eines Sprachassistenten, der ohne die Anbindung an Cloud Dienste funktioniert.

## TODOs
- Implement multi language support
- In case that none of the intents fits a request, add GPT-2 to provide an answer instead of an "I did not understand".
- Port to RasperryPi
- Highlight, whenever the assistant requires an online connection (e.g. via an LED or a popup)
- Show how to train speech-2-text (e.g. via speechbrain)
- Show how to train text-2-speech (CorentinJ)
- Introduce pyflakes to remove unused packages
- Show how to create a cluster of multiple assistants, e.g. for each room
- Add more interactivity, let the assistant initiate a conversation ("how was your day?", "you asked for the weather yesterday, did you like it?")