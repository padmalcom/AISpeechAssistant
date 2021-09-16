import spacy
from spacy.util import minibatch, compounding
import time
import random
from sklearn.metrics import classification_report
import os
import string
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

#https://github.com/rsreetech/TextClassificationWithSpacy/blob/master/TweetTextClassificationWithSpacy.ipynb
def Sort(sub_li): 
	return(sorted(sub_li, key = lambda x: x[1],reverse=True))  

# run the predictions on each sentence in the evaluation  dataset, and return the metrics
def evaluate(tokenizer, textcat, test_texts, test_cats):
	docs = (tokenizer(text) for text in test_texts)
	preds = []
	for i, doc in enumerate(textcat.pipe(docs)):
		scores = Sort(doc.cats.items())
		catList=[]
		for score in scores:
			catList.append(score[0])
		preds.append(catList[0])
		
	labels = ["AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook", "SearchCreativeWork", "SearchScreeningEvent"]
	
	print(classification_report(test_cats, preds, labels=labels))
	
def load():
	df_data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.csv"))
	
	df_data = shuffle(df_data)
	df_data.reset_index(inplace=True, drop=True)
	
	train, test = np.split(df_data, [int(len(df_data)*0.8)])
		
	train_set = {'sentences': train.text_de.tolist(), 'intents': train.intent_index.tolist()}
	test_set = {'sentences': test.text_de.tolist(), 'intents': test.intent.tolist()}
	
	return train_set, test_set

def train(train_data, iterations, test_texts,test_cats, model_arch, dropout = 0.3, model=None,init_tok2vec=None):
	nlp = spacy.load("de_core_news_sm")
	
	# add the text classifier to the pipeline if it doesn't exist
	textcat = nlp.create_pipe("textcat", config={"exclusive_classes": True, "architecture": model_arch})
	nlp.add_pipe(textcat, last=True)

	# add label to text classifier
	textcat.add_label("AddToPlaylist")
	textcat.add_label("BookRestaurant")
	textcat.add_label("GetWeather")
	textcat.add_label("PlayMusic")
	textcat.add_label("RateBook")
	textcat.add_label("SearchCreativeWork")
	textcat.add_label("SearchScreeningEvent")

	# get names of other pipes to disable them during training
	pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
	other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
	with nlp.disable_pipes(*other_pipes):  # only train textcat
		optimizer = nlp.begin_training()
		if init_tok2vec is not None:
			with init_tok2vec.open("rb") as file_:
				textcat.model.tok2vec.from_bytes(file_.read())
		print("Training the model...")
		#print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
		batch_sizes = compounding(16.0, 64.0, 1.5)
		for i in range(iterations):
			print('Iteration: '+str(i))
			start_time = time.perf_counter()
			losses = {}
			# batch up the examples using spaCy's minibatch
			random.shuffle(train_data)
			batches = minibatch(train_data, size=batch_sizes)
			for batch in batches:
				texts, annotations = zip(*batch)
				nlp.update(texts, annotations, sgd=optimizer, drop=dropout, losses=losses)
			with textcat.model.use_params(optimizer.averages):
				# evaluate on the test data 
				evaluate(nlp.tokenizer, textcat, test_texts, test_cats)
			print ('Elapsed time '+str(time.perf_counter() - start_time)+  " seconds")
		with nlp.use_params(optimizer.averages):
			modelName = model_arch+"_model"
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
	
	nlp = train(training_data, 10, test_texts, test_cats, "ensemble")
	
	test1 = "Buche ein Restaurant für heute abend".lower()
	test2 = "Füge alle Musik von Blink 182 meiner Playlist hinzu".lower()
	nlp2 = spacy.load(os.path.join(os.getcwd(), "ensemble_model"))
	doc2 = nlp2(test1)
	print("Text: "+ test1)
	print(doc2.cats)
	
	doc3 = nlp2(test2)
	print("Text: "+ test2)
	print(doc3.cats)
