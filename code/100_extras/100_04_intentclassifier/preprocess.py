import glob, os, json
import csv
import numpy as np
from chardet import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random

# Data source https://github.com/sonos/nlu-benchmark/tree/master/2017-06-custom-intent-engines

# prepare models for translation. Replace with any other model if required.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer_en_de = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
model_en_de = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de").to(device)
tokenizer_en_fr = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
model_en_fr = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr").to(device)
tokenizer_en_es = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
model_en_es = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es").to(device)

# use transformers for translation task
def translate_en_to_de(text):
	with tokenizer_en_de.as_target_tokenizer():
		tokenized_text = tokenizer_en_de(text, return_tensors='pt').to(device)
	translation = model_en_de.generate(**tokenized_text)
	return tokenizer_en_de.batch_decode(translation, skip_special_tokens=True)[0]
	
def translate_en_to_fr(text):
	with tokenizer_en_fr.as_target_tokenizer():
		tokenized_text = tokenizer_en_fr(text, return_tensors='pt').to(device)
	translation = model_en_fr.generate(**tokenized_text)
	return tokenizer_en_fr.batch_decode(translation, skip_special_tokens=True)[0]
	
def translate_en_to_es(text):
	with tokenizer_en_es.as_target_tokenizer():
		tokenized_text = tokenizer_en_es(text, return_tensors='pt').to(device)
	translation = model_en_es.generate(**tokenized_text)
	return tokenizer_en_es.batch_decode(translation, skip_special_tokens=True)[0]
	
# get the encoding of a file using chardet
def get_encoding_type(file):
	with open(file, 'rb') as f:
		rawdata = f.read()
	return detect(rawdata)['encoding']
	
# execute the preprocessing. When using translation, this will take some hours
if __name__ == '__main__':
	
	current_dir = os.path.dirname(os.path.abspath(__file__))
	
	# Open an individual train, test and validation data file
	with open(os.path.join(current_dir, 'train.csv'), 'w', encoding='utf-8', newline='') as train_file, open(os.path.join(current_dir, 'test.csv'), 'w', encoding='utf-8', newline='') as test_file, open(os.path.join(current_dir, 'validation.csv'), 'w', encoding='utf-8', newline='') as validation_file:
			train_csv_writer = csv.writer(train_file)
			test_csv_writer = csv.writer(test_file)
			validation_csv_writer = csv.writer(validation_file)
			
			# write headers
			train_csv_writer.writerow(['text_en', 'text_de', 'text_fr', 'text_es', 'intent', 'intent_index'])
			test_csv_writer.writerow(['text_en', 'text_de', 'text_fr', 'text_es', 'intent', 'intent_index'])
			validation_csv_writer.writerow(['text_en', 'text_de', 'text_fr', 'text_es', 'intent', 'intent_index'])
			
			# Read all JSON files
			all_texts = []
			for file in glob.glob(os.path.join(current_dir, "data", "*_full.json")):
			
				# Detect the file encoding
				detected_encoding = get_encoding_type(file)
				intent_index = 0
				with open(file, 'r', encoding=detected_encoding, errors='ignore') as json_file:
					data = json.load(json_file)
					title = list(data.keys())[0]
					print("Processing intent " + title + " ...")
					
					
					length = len(data[title])
					for index, entry in enumerate(data[title]):
						print("Processing entry " + str(index) + " of " + str(length) + " ...")
						intent_text = ""
						texts = entry['data']
						for text in texts:
							intent_text += text['text']
						translated_de = translate_en_to_de(intent_text)
						translated_fr = translate_en_to_fr(intent_text)
						translated_es = translate_en_to_es(intent_text)
						all_texts.append({'text_en': intent_text, 'text_de': translated_de, 'text_fr': translated_fr, 'text_es': translated_es, 'intent': title, 'intent_index': intent_index})
					print("Data for ", title, ":", length)
					intent_index += 1

			# Shuffle entire data
			random.shuffle(all_texts)
			
			# Split at 60% and 80%, so that ratio = 60, 20, 20
			train, validate, test = np.split(all_texts, [int(len(all_texts)*0.6), int(len(all_texts)*0.8)])
			
			
			for t in train:
				train_csv_writer.writerow([t['text_en'].strip(), t['text_de'].strip(), t['text_fr'].strip(), t['text_es'].strip(), t['intent'].strip(), t['intent_index']])
				
			for t in test:
				test_csv_writer.writerow([t['text_en'].strip(), t['text_de'].strip(), t['text_fr'].strip(), t['text_es'].strip(), t['intent'].strip(), t['intent_index']])
				
			for v in validate:
				validation_csv_writer.writerow([v['text_en'].strip(), v['text_de'].strip(), v['text_fr'].strip(), v['text_es'].strip(), v['intent'].strip(), v['intent_index']])
			print("Train size: ", len(train), " test size: ", len(validate), "validation size: ", len(test))
					