# src: https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b
import sys
import os

from loguru import logger
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils import shuffle
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from spacy.lang.de import German

tqdm.pandas()

DO_TRAIN = False
DO_PREDICT = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Read data (or create if not existent)
# Data src: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
if not os.path.exists(os.path.join('data', 'train.csv')):
	fake_data = pd.read_csv("data/Fake.csv")
	true_data = pd.read_csv("data/True.csv")

	fake_data['fake'] = True
	true_data['fake'] = False

	train_data = pd.concat([fake_data, true_data])
	#train_data = train_data.head(20000) # for dev purposes
	
	# translate
	tokenizer = FSMTTokenizer.from_pretrained("facebook/wmt19-en-de")
	model = FSMTForConditionalGeneration.from_pretrained("facebook/wmt19-en-de").to(device)

	# Cut into sentences
	nlp = German()
	nlp.add_pipe('sentencizer')
		
	def translate_en_to_de(text):
		doc = nlp(text)
		sentences = [sent.text.strip() for sent in doc.sents]
		logger.info("Text has been split into {} sentences.", len(sentences))
		trans_text = ""
		sentence_to_process = ""
		for idx, sentence in enumerate(sentences):
			
			if len(sentence_to_process) + len(sentence) > 1024:
				input_ids = tokenizer.encode(sentence_to_process.strip(), return_tensors="pt").to(device)
				outputs = model.generate(input_ids)
				trans_text += tokenizer.decode(outputs[0], skip_special_tokens=True) + " "
				sentence_to_process = sentence
			elif idx == len(sentences)-1:
				# last sentence
				sentence_to_process += " " + sentence
				input_ids = tokenizer.encode(sentence_to_process.strip(), return_tensors="pt").to(device)
				outputs = model.generate(input_ids)
				trans_text += tokenizer.decode(outputs[0], skip_special_tokens=True) + " "
			else:
				sentence_to_process += " " + sentence
		return trans_text
	
	train_data['text'] = train_data['text'].progress_apply(translate_en_to_de)
	train_data['title'] = train_data['title'].progress_apply(translate_en_to_de)	
	
	train_data.to_csv('train.csv', index=False)
else:
	train_data = pd.read_csv(os.path.join('data', 'train.csv'))

# Define pretrained tokenizer and model
model_name = "deepset/gbert-base"
output_dir = "output"
tokenizer = BertTokenizer.from_pretrained(model_name)

if DO_TRAIN == True:	
	train_data = shuffle(train_data)
	train_data, test_data = np.split(train_data, [int(.8*len(train_data))])
	logger.info("Length train data {}, length test data {}.", len(train_data), len(test_data))




	# Lade den letzten Checkpoint
	last_checkpoint = get_last_checkpoint(output_dir)
	if last_checkpoint is not None:
		checkpoint = last_checkpoint
	elif os.path.isdir(model_name):
		checkpoint = model_args.model_name_or_path
	else:
		checkpoint = None
	logger.info("Letzter Trainings-Checkpoint: {}", checkpoint)
		
	
	model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

	# ----- 1. Preprocess data -----#
	# Preprocess data

	train_data["fake"] = train_data["fake"].apply(lambda x: 1 if x == True else 0)
	X = list(train_data["text"])
	y = list(train_data["fake"])
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
	X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
	X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

	# Create torch dataset
	class Dataset(torch.utils.data.Dataset):
		def __init__(self, encodings, labels=None):
			self.encodings = encodings
			self.labels = labels

		def __getitem__(self, idx):
			item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
			if self.labels:
				item["labels"] = torch.tensor(self.labels[idx])
			return item

		def __len__(self):
			return len(self.encodings["input_ids"])

	train_dataset = Dataset(X_train_tokenized, y_train)
	val_dataset = Dataset(X_val_tokenized, y_val)

	print(train_dataset)
	print(val_dataset)

# ----- 2. Fine-tune pretrained model -----#

	# Define Trainer parameters
	def compute_metrics(p):
		pred, labels = p
		pred = np.argmax(pred, axis=1)

		accuracy = accuracy_score(y_true=labels, y_pred=pred)
		recall = recall_score(y_true=labels, y_pred=pred)
		precision = precision_score(y_true=labels, y_pred=pred)
		f1 = f1_score(y_true=labels, y_pred=pred)

		return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

	# Define Trainer
	args = TrainingArguments(
		output_dir=output_dir,
		evaluation_strategy="steps",
		eval_steps=500,
		per_device_train_batch_size=8,
		per_device_eval_batch_size=8,
		num_train_epochs=3,
		seed=0,
		load_best_model_at_end=True,
	)
	trainer = Trainer(
		model=model,
		args=args,
		train_dataset=train_dataset,
		eval_dataset=val_dataset,
		compute_metrics=compute_metrics,
		callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
	)

	# Train pre-trained model
	trainer.train()

if DO_PREDICT == True:
	# Load trained model
	model_path = "output/checkpoint-4000"
	model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2).to(device)


	model.eval()
	inputs = tokenizer("Obama ist nicht in den USA geboren.", padding=True, truncation=True, return_tensors="pt").to(device)
	with torch.no_grad():
		outputs = model(**inputs)
		probs = outputs[0].softmax(1)
		target_names = [False, True]
		is_fake = target_names[probs.argmax()]
		logger.info("Diese Aussage ist: {}", is_fake)


	
