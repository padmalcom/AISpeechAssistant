import spacy
from spacy.util import minibatch, compounding
import time, random, os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

# Es werden Vorhersagen für den Evaluierungsdatensatz getroffen und die
# Metriken ausgegeben.
def evaluate(tokenizer, textcat, test_texts, test_cats):
	docs = (tokenizer(text) for text in test_texts)
	preds = []
	for i, doc in enumerate(textcat.pipe(docs)):
		scores = sorted(doc.cats.items(), key = lambda x: x[1],reverse=True)
		catList=[]
		for score in scores:
			catList.append(score[0])
		preds.append(catList[0])
		
	labels = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]
	cm = confusion_matrix(test_cats, preds, labels=labels)
	ConfusionMatrixDisplay(cm, display_labels=labels).plot()
	plt.xticks(rotation=90)
	plt.xlabel("Vorhergesagte Klasse")
	plt.ylabel("Tatsächliche Klasse")
	plt.show()
	
	print(classification_report(test_cats, preds, labels=labels))
	
def load():
	df_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.csv"))
	
	df_data = shuffle(df_data)
	df_data.reset_index(inplace=True, drop=True)
	
	train, test = np.split(df_data, [int(len(df_data)*0.8)])
		
	train_set = {'sentences': train.text_de.tolist(), 'intents': train.intent_index.tolist()}
	test_set = {'sentences': test.text_de.tolist(), 'intents': test.intent.tolist()}
	
	return train_set, test_set

def train(train_data, iterations, test_texts, test_cats, dropout = 0.3):
	nlp = spacy.load("de_core_news_sm")
	
	textcat = nlp.create_pipe("textcat", config={"exclusive_classes": True, "architecture": "ensemble"})
	nlp.add_pipe(textcat, last=True)

	# Festlegen der Intents als Labels für den Classifier
	textcat.add_label("AddToPlaylist")
	textcat.add_label("BookRestaurant")
	textcat.add_label("GetWeather")
	textcat.add_label("PlayMusic")
	textcat.add_label("RateBook")
	textcat.add_label("SearchCreativeWork")
	textcat.add_label("SearchScreeningEvent")

	# Nur die Pipeline für den Text Classifier wird benötigt.
	pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
	other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
	with nlp.disable_pipes(*other_pipes):
		optimizer = nlp.begin_training()

		print("Beginne Training ...")
		#batch_sizes = compounding(16.0, 64.0, 1.5)
		for i in range(iterations):
			print('Iteration: '+str(i))
			start_time = time.perf_counter()
			losses = {}
			random.shuffle(train_data)
			#batches = minibatch(train_data, size=batch_sizes)
			batches = minibatch(train_data, size=16)
			for batch in batches:
				texts, annotations = zip(*batch)
				nlp.update(texts, annotations, sgd=optimizer, drop=dropout, losses=losses)
			with textcat.model.use_params(optimizer.averages):
				evaluate(nlp.tokenizer, textcat, test_texts, test_cats)
				
			print ("Iteration " + str(i) + ": " +str(time.perf_counter() - start_time)+  " Sekunden")
			
		with nlp.use_params(optimizer.averages):
			modelName = "ensemble_model"
			filepath = os.path.join(os.getcwd(), modelName)
			nlp.to_disk(filepath)
	return nlp
	
if __name__ == '__main__':
	train_set, test_set = load()
	train_texts = train_set['sentences'] 
	train_cats = train_set['intents']
		
	final_train_cats=[]
	for cat in train_cats:

		cat_list = {'AddToPlaylist': 1 if cat == 0 else 0,
			'BookRestaurant': 1 if cat == 1 else 0,
			'GetWeather': 1 if cat == 2 else 0,
			'PlayMusic': 1 if cat == 3 else 0,
			'RateBook': 1 if cat == 4 else 0,
			'SearchCreativeWork': 1 if cat == 5 else 0,
			'SearchScreeningEvent': 1 if cat == 6 else 0,
		}
		final_train_cats.append(cat_list)
	
	training_data = list(zip(train_texts, [{"cats": cats} for cats in final_train_cats]))
	
	test_texts = test_set['sentences'] 
	test_cats = test_set['intents']
	
	nlp = train(training_data, 10, test_texts, test_cats)
	
	test1 = "Buche ein Restaurant für heute abend".lower()
	test2 = "Füge alle Musik von Blink 182 meiner Playlist hinzu".lower()
	nlp2 = spacy.load(os.path.join(os.getcwd(), "ensemble_model"))
	doc2 = nlp2(test1)
	print("Text: "+ test1)
	print(doc2.cats)
	
	doc3 = nlp2(test2)
	print("Text: "+ test2)
	print(doc3.cats)
