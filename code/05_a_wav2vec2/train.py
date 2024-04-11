from dataclasses import dataclass
from transformers import (
	TrainingArguments,
	HfArgumentParser,
	Wav2Vec2FeatureExtractor,
	Wav2Vec2CTCTokenizer,
	Wav2Vec2Processor
)
import datasets
import evaluate
import pandas as pd
import re
import librosa
import os
import torch
import numpy as np
import json
from model import Wav2Vec2ForCTCnCLS
from ctctrainer import CTCTrainer
from datacollator import DataCollatorCTCWithPadding
from tokenizer import build_tokenizer

# Feature order: age, gender, emotion, dialect

@dataclass
class DataTrainingArguments:
	target_text_column = "sentence"
	speech_file_column = "file"
	age_column = "age"
	gender_column = "gender"
	emotion_column = "emotion"
	dialect_column = "accent"
	preprocessing_num_workers = 1
	output_dir = "output/tmp"
	
@dataclass
class ModelArguments:
	model_name_or_path = "facebook/wav2vec2-base"
	#model_name_or_path = "facebook/wav2vec2-large-xlsr-53"
	cache_dir = "cache/"
	freeze_feature_extractor = True
	alpha = 0.1
					
if __name__ == "__main__":
	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
	model_args, data_args, training_args = parser.parse_args_into_dataclasses()
	
	os.makedirs(training_args.output_dir, exist_ok=True)
		
	base_path = os.path.join('D:', os.sep, 'Datasets', 'common-voice-16-full')
		
	# Load dataset
	dataset = datasets.load_dataset('csv', data_files={'train': os.path.join(base_path, 'train.csv'), 'test': os.path.join(base_path, 'test.csv')},
        cache_dir="G:\\datasets\\")
	print("Dataset:", dataset)
	print("Test:", dataset['test'])
	print("Test0:", dataset['test'][0])
			
	german_char_map = {ord('ä'):'ae', ord('ü'):'ue', ord('ö'):'oe', ord('ß'):'ss', ord('Ä'): 'Ae', ord('Ü'):'Ue', ord('Ö'):'Oe'}
	
	def remove_special_characters(batch):
		batch["sentence"] = batch["sentence"].translate(german_char_map)
		batch["sentence"] = batch["sentence"].encode('ascii', errors='ignore')
		return batch
		
	dataset = dataset.map(remove_special_characters)
	
	# create processor
	feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
	
	# create and save tokenizer
	tokenizer = build_tokenizer(training_args.output_dir, dataset)
	tokenizer.save_pretrained(os.path.join(training_args.output_dir, "tokenizer"))
	
	# create processor
	processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
	print("vocab size: ", len(processor.tokenizer))
		
	# create label maps and count of each label class	
	cls_age_label_map = {'teens':0, 'twenties': 1, 'thirties': 2, 'fourties': 3, 'fifties': 4, 'sixties': 5, 'seventies': 6, 'eighties': 7, 'nineties': 8}
	cls_age_class_weights = [0] * len(cls_age_label_map)
	
	cls_gender_label_map = {'female': 0, 'male': 1, 'other': 2}
	cls_gender_class_weights = [0] * len(cls_gender_label_map)
	
	cls_emotion_label_map = {'anger':0, 'boredom':1, 'disgust':2, 'fear':3, 'happiness':4, 'sadness':5, 'neutral':6}
	cls_emotion_class_weights = [0] * len(cls_emotion_label_map)
	
	cls_dialect_label_map = {'Französisch': 0,'Russisch': 1,'Schweizerdeutsch': 2,'Deutsch': 3,'Österreichisch': 4,'Polnisch': 5,'Amerikanisch': 6,'Dänisch': 7,
        'Türkisch': 8,'Britisch': 9,'Tschechisch': 10,'Niederländisch': 11,'Italienisch': 12,'Griechisch': 13,'Hessisch': 14,'Bayrisch': 15,'Ungarisch': 16,
        'Bulgarisch': 17,'Belgisch': 18,'Arabisch': 19,'Slowakisch': 20,'Süddeutsch': 21,'Litauisch': 22,'Schwäbisch': 23,'Sächsisch': 24,'Luxemburgisch': 25,
        'Rheinländisch': 26,'Kanadisch': 27,'Badisch': 28,'Israelisch': 29,'Lichtensteinisch': 30,'Slowenisch': 31,'Brasilianisch': 32,'Saarländisch': 33,
        'Ruhrdeutsch': 34,'Finnisch': 35,'Fränkisch': 36,'Berlinerisch': 37,'Lettisch': 38,'Niederrheinisch': 39}
	cls_dialect_class_weights = [0] * len(cls_dialect_label_map)
	
	# count label sizes in train to balance classes
	df = pd.read_csv(os.path.join(base_path, 'train.csv'))
	
	df_age_count = df.groupby(['age']).count()
	for index, k in enumerate(cls_age_label_map):
		if k in df_age_count.index:
			cls_age_class_weights[index] = 1 - (df_age_count.loc[k]['file'] / df.size)
	print("Age label weights:", cls_age_class_weights)
	
	df_gender_count = df.groupby(['gender']).count()
	for index, k in enumerate(cls_gender_label_map):
		if k in df_gender_count.index:
			cls_gender_class_weights[index] = 1 - (df_gender_count.loc[k]['file'] / df.size)
	print("Gender label weights:", cls_gender_class_weights)
	
	df_emotion_count = df.groupby(['emotion']).count()
	for index, k in enumerate(cls_emotion_label_map):
		if k in df_emotion_count.index:
			cls_emotion_class_weights[index] = 1 - (df_emotion_count.loc[k]['file'] / df.size)
	print("Emotion label weights:", cls_emotion_class_weights)	
	
	df_dialect_count = df.groupby(['accent']).count()
	for index, k in enumerate(cls_dialect_label_map):
		if k in df_dialect_count.index:
			cls_dialect_class_weights[index] = 1 - (df_dialect_count.loc[k]['file'] / df.size)
	print("Dialect label weights:", cls_dialect_class_weights)
	
	# Load model
	model = Wav2Vec2ForCTCnCLS.from_pretrained(
		model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		gradient_checkpointing=True,
		vocab_size=len(processor.tokenizer),
		age_cls_len=len(cls_age_label_map),
		age_cls_weights=cls_age_class_weights,
		gender_cls_len=len(cls_gender_label_map),
		gender_cls_weights=cls_gender_class_weights,
		emotion_cls_len=len(cls_emotion_label_map),
		emotion_cls_weights=cls_emotion_class_weights,
		dialect_cls_len=len(cls_dialect_label_map),
		dialect_cls_weights=cls_dialect_class_weights,		
		alpha=model_args.alpha,
	)
	
	# load metrics
	wer_metric = evaluate.load("wer")
	
	# preprocess data
	target_sr = 16000
	vocabulary_chars_str = "".join(t for t in processor.tokenizer.get_vocab().keys() if len(t) == 1)
	vocabulary_text_cleaner = re.compile(
		f"[^\s{re.escape(vocabulary_chars_str)}]",
		flags=re.IGNORECASE if processor.tokenizer.do_lower_case else 0,
	)
	
	def prepare_example(example, audio_only=False):
		example["speech"], example["sampling_rate"] = librosa.load(os.path.join(base_path, "wavs", example[data_args.speech_file_column]), sr=target_sr)
		if audio_only is False:
			updated_text = " ".join(example[data_args.target_text_column].split()) # remove whitespaces
			updated_text = vocabulary_text_cleaner.sub("", updated_text)
			if updated_text != example[data_args.target_text_column]:
				example[data_args.target_text_column] = updated_text
		return example
		
	train_dataset = dataset.map(prepare_example, remove_columns=[data_args.speech_file_column])['train']
	val_dataset = dataset.map(prepare_example, remove_columns=[data_args.speech_file_column])['test']
	
	print("train:", train_dataset)
	print("eval:", val_dataset)
	
	def prepare_dataset(batch, audio_only=False):
		batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
		if audio_only is False:
			age_cls_labels = list(map(lambda e: cls_age_label_map[e], batch[data_args.age_column]))
			gender_cls_labels = list(map(lambda e: cls_gender_label_map[e], batch[data_args.gender_column]))
			emotion_cls_labels = list(map(lambda e: cls_emotion_label_map[e], batch[data_args.emotion_column]))
			dialect_cls_labels = list(map(lambda e: cls_dialect_label_map[e], batch[data_args.dialect_column]))
			with processor.as_target_processor():
				batch["labels"] = processor(batch[data_args.target_text_column]).input_ids

			# attention to inverse order!
			for i in range(len(dialect_cls_labels)):
				batch["labels"][i].append(dialect_cls_labels[i]) # batch["labels"] element has to be a single list

			for i in range(len(emotion_cls_labels)):
				batch["labels"][i].append(emotion_cls_labels[i]) # batch["labels"] element has to be a single list	
				
			for i in range(len(gender_cls_labels)):
				batch["labels"][i].append(gender_cls_labels[i]) # batch["labels"] element has to be a single list

			for i in range(len(age_cls_labels)):
				batch["labels"][i].append(age_cls_labels[i]) # batch["labels"] element has to be a single list

		# the last items in the labels list are: gender label and age label
		return batch
		
	train_dataset = train_dataset.map(
		prepare_dataset,
		batch_size=training_args.per_device_train_batch_size,
		batched=True,
		num_proc=data_args.preprocessing_num_workers,
	)
	
	val_dataset = val_dataset.map(
		prepare_dataset,
		batch_size=training_args.per_device_train_batch_size,
		batched=True,
		num_proc=data_args.preprocessing_num_workers,
	)
	
	data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
	
	def compute_metrics(pred):	
		age_id = 1
		gender_id = 2
		emotion_id = 3
		dialect_id = 4
		age_cls_pred_logits = pred.predictions[age_id]
		age_cls_pred_ids = np.argmax(age_cls_pred_logits, axis=-1)
		age_total = len(pred.label_ids[age_id])
		age_correct = (age_cls_pred_ids == pred.label_ids[age_id]).sum().item()
		
		gender_cls_pred_logits = pred.predictions[gender_id]
		gender_cls_pred_ids = np.argmax(gender_cls_pred_logits, axis=-1)
		gender_total = len(pred.label_ids[gender_id])
		gender_correct = (gender_cls_pred_ids == pred.label_ids[gender_id]).sum().item()
		
		emotion_cls_pred_logits = pred.predictions[emotion_id]
		emotion_cls_pred_ids = np.argmax(emotion_cls_pred_logits, axis=-1)
		emotion_total = len(pred.label_ids[emotion_id])
		emotion_correct = (emotion_cls_pred_ids == pred.label_ids[emotion_id]).sum().item()		

		dialect_cls_pred_logits = pred.predictions[dialect_id]
		dialect_cls_pred_ids = np.argmax(dialect_cls_pred_logits, axis=-1)
		dialect_total = len(pred.label_ids[dialect_id])
		dialect_correct = (dialect_cls_pred_ids == pred.label_ids[dialect_id]).sum().item()

		ctc_pred_logits = pred.predictions[0]
		ctc_pred_ids = np.argmax(ctc_pred_logits, axis=-1)
		pred.label_ids[0][pred.label_ids[0] == -100] = processor.tokenizer.pad_token_id
		ctc_pred_str = processor.batch_decode(ctc_pred_ids)
		# we do not want to group tokens when computing the metrics
		ctc_label_str = processor.batch_decode(pred.label_ids[0], group_tokens=False)


		wer = wer_metric.compute(predictions=ctc_pred_str, references=ctc_label_str)
		accuracy = ((age_correct / age_total) + (gender_correct / gender_total) + (emotion_correct / emotion_total) + (dialect_correct / dialect_total)) / 4
		print("Age correct:", age_correct, "of total:", age_total, "accuracy: ", (age_correct / age_total))
		print("Gender correct:", gender_correct, "of total:", gender_total, "accuracy: ", (gender_correct / gender_total))
		print("Emotion correct:", emotion_correct, "of total:", emotion_total, "accuracy: ", (emotion_correct / emotion_total))		
		print("Dialect correct:", dialect_correct, "of total:", dialect_total, "accuracy: ", (dialect_correct / dialect_total))
		
		metric_res = {"acc": accuracy, "wer": wer, "correct": age_correct + gender_correct + emotion_correct + dialect_correct,
			"total": age_total + gender_total + emotion_total + dialect_total, "strlen": len(ctc_label_str)}
		print("metric result:", metric_res)
		return metric_res
		
	if model_args.freeze_feature_extractor:
		model.freeze_feature_extractor()
		
	print("Val dataset:", val_dataset)
	print("Train dataset:", train_dataset)
	trainer = CTCTrainer(
		model=model,
		data_collator=data_collator,
		args=training_args,
		compute_metrics=compute_metrics,
		train_dataset=train_dataset,
		eval_dataset=val_dataset,
		tokenizer=processor.feature_extractor
	)
	trainer.train()
	trainer.save_model(training_args.output_dir) 