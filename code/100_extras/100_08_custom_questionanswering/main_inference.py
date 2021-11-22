from transformers import pipeline
from multiprocessing import freeze_support

if __name__ == '__main__':
	freeze_support()
	qa_pipeline = pipeline(
		"question-answering",
		model="./models",
		tokenizer="./models"
	)
	
	contexts = ['''Obamas Vater, Barack Hussein Obama Senior (1936–1982), stammte aus Nyang’oma Kogelo in Kenia und gehörte der Ethnie der Luo an. Obamas Mutter, Stanley Ann Dunham (1942–1995), stammte aus Wichita im US-Bundesstaat Kansas und hatte irische, britische, deutsche und Schweizer Vorfahren.[6] Obamas Eltern lernten sich als Studenten an der University of Hawaii at Manoa kennen. Sie heirateten 1961 in Hawaii, als Ann bereits schwanger war. Damals waren in anderen Teilen der USA Ehen zwischen Schwarzen und Weißen noch verboten. 1964 ließ sich das Paar scheiden. Der Vater setzte sein Studium an der Harvard University fort. Obama sah ihn als Zehnjähriger zum letzten Mal.''']*2

	questions = ["Woher kommt Obamas Vater?", 
				"Wann sah Obama seinen Vater zum letzten Mal?"]

	print(qa_pipeline(context=contexts, question=questions))