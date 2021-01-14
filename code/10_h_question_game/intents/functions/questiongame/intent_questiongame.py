from chatbot import register_call
import random
import sys
import os
import global_variables

YES = ["JA", "J", "YES", "Y"]
NO = ["NEIN", "N", "NO"]
PROBABLY = ["VIELLEICHT", "PROBABLY"]
PROBABLY_NOT = ["WAHRSCHEINLICH NICHT", "EHER NICHT", "PROBABLY NOT"]

DID_NOT_UNDESTAND = "Ich habe die Antwort nicht verstanden"

question_game_session = None

@register_call("startQuestionGame")
def startQuestionGame(session_id = "general", dummy=0):
	print("Starte neues Spiel")
	global question_game_session
	question_game_session = Q20Session()
	global_variables.context = questionGameAnswer
	return question_game_session.askQuestion()

def questionGameAnswer(answer=""):
	print("Antwort " + str(answer) + " erhalten.")
	global question_game_session
	if not question_game_session is None:
		print("Session existiert")
		answer_value = question_game_session.evaluateAnswer(answer)
		for i in range(len(question_game_session.items)):
			question_game_session.items[i].updateCertainty(answer_value, len(question_game_session.questions))			
		question = question_game_session.askQuestion()
		if question:
			print("Frage gefunden: " + str(question))
			global_variables.reset_session = False
			return question
		else:
			print("Keine Frage gefunden.")
			final_answer = question_game_session.getAnswer()
			print("Antwort ist " + str(final_answer))
			question_game_session.clearSession()
			global_variables.reset_session = True
			question_game_session = None
			print("No more session")
			return final_answer
	else:
		return "Bitte starte neues Spiel mit 'Starte Fragespiel'"

class Q20Session():
	
	def __init__(self):
		self.questions = []
		self.items = []
		#self.guessed = False
		self.current_question = 0
		
		items_path = os.path.join('intents','functions','questiongame', 'items.txt')
		questions_path = os.path.join('intents','functions','questiongame', 'questions.txt')
		
		itemData=open(items_path, encoding="utf-8")
		data=itemData.readlines()
		for i in range(len(data)):
			subdata=data[i].rstrip("\n").split(":")
			questionFloats=subdata[2][1:-1].split(",")
			for i in range(len(questionFloats)):
				questionFloats[i]=round(float(questionFloats[i]),4)
			self.items.append(Item(subdata[0],int(subdata[1]),questionFloats, len(self.items)))
		questionData=open(questions_path, encoding="utf-8")
		data=questionData.readlines()
		for i in range(len(data)):
			self.questions.append(Question(data[i].rstrip("\n"), len(self.questions)))
		questionData.close()
		
	def askQuestion(self):
		if self.current_question < len(self.questions):
			question = self.questions[self.current_question].string
			self.current_question += 1
			return question
		else:
			return None
			
	def getAnswer(self):
		selectedData = self.evaluateCertainties()
		result = "Mit einer Wahrscheinlichkeit von " + str(round(100*selectedData[1],2)) + " Prozent denkst du an den Begriff " + self.items[selectedData[0]].name.capitalize()
		return result
			
	def evaluateAnswer(self, answer):
		if answer.upper() in YES:
			a = 1
		elif answer.upper() in NO:
			a = -1
		elif answer.upper() in PROBABLY:
			a = 0.2
		elif answer.upper() in PROBABLY_NOT:
			a = -0.2
		else:
			a = 0
		return a
		
	def evaluateCertainties(self):
		maxi=0
		selected=-1
		for i in range(len(items)):
			if items[i].certainty>maxi:
				maxi=items[i].certainty
				selected=i
		return [selected, maxi]

	def clearSession(self):
		#self.answers=[]
		#self.guessed=False
		for i in range(len(self.items)):
			self.items[i].certainty=0

class Question():
	def __init__(self, string, id):
		self.string=string
		self.index=id  
#	def ask(self):
		#global items, answers
#		print(self.string)
#		a=getInput()
#		for i in range(len(items)):
#			items[i].updateCertainty(a)
		#answers.append(a)
		
class Item():
	def __init__(self, name, guessNum, questionFloats, id):
		self.name=name
		self.guessNum=guessNum
		self.questionFloats=questionFloats
		self.index=id
		self.certainty=0
		
	def updateCertainty(self, val, num_questions):
		self.certainty+=(1-abs(val-self.questionFloats[num_questions-1]))/num_questions
			
if __name__ == '__main__':
	session = Q20Session()
	session.run()