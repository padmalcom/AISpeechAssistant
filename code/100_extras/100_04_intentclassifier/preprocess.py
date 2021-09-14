import glob, os, json
import csv
import numpy as np
from chardet import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Data source https://github.com/sonos/nlu-benchmark/tree/master/2017-06-custom-intent-engines

# prepare models for translation. Replace with any other model if required.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de").to(device)

# use transformers for translation task
def translate_en_to_de(text):
	with tokenizer.as_target_tokenizer():
		tokenized_text = tokenizer(text, return_tensors='pt').to(device)
	translation = model.generate(**tokenized_text)
	return tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
	
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
			train_csv_writer.writerow(['text', 'intent'])
			test_csv_writer.writerow(['text', 'intent'])
			validation_csv_writer.writerow(['text', 'intent'])
			
			# Read all JSON files
			for file in glob.glob(os.path.join(current_dir, "data", "*_full.json")):
			
				# Detect the file encoding
				detected_encoding = get_encoding_type(file)
				with open(file, 'r', encoding=detected_encoding, errors='ignore') as json_file:
					data = json.load(json_file)
					title = list(data.keys())[0]
					print("Processing intent " + title + " ...")
					
					all_texts = []
					length = len(data[title])
					for index, entry in enumerate(data[title]):
						print("Processing entry " + str(index) + " of " + str(length) + " ...")
						intent_text = ""
						texts = entry['data']
						for text in texts:
							intent_text += text['text']
						translated_text = translate_en_to_de(intent_text)
						all_texts.append({'text': translated_text, 'intent': title})
					print("Data for ", title, ":", len(all_texts))
					
					# Split at 60% and 80%, so that ratio = 60, 20, 20
					train, validate, test = np.split(all_texts, [int(len(all_texts)*0.6), int(len(all_texts)*0.8)])
					
					for t in train:
						train_csv_writer.writerow([t['text'].strip(), t['intent'].strip()])
						
					for t in test:
						test_csv_writer.writerow([t['text'].strip(), t['intent'].strip()])
						
					for v in validate:
						validation_csv_writer.writerow([v['text'].strip(), v['intent'].strip()])
					print("Train size: ", len(train), " test size: ", len(validate), "validation size: ", len(test))