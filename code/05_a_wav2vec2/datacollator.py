import torch

class DataCollatorCTCWithPadding:
	def __init__(self, processor, padding, audio_only=False):
		self.processor = processor
		self.padding = padding
		self.max_length = None
		self.max_length_labels = None
		self.pad_to_multiple_of = None
		self.pad_to_multiple_of_labels = None
		self.audio_only = audio_only

	def __call__(self, features):
		# split inputs and labels since they have to be of different lenghts and need
		# different padding methods
		input_features = [{"input_values": feature["input_values"]} for feature in features]
		if self.audio_only is False:
			label_features = [{"input_ids": feature["labels"][:-4]} for feature in features]
			age_cls_labels = [feature["labels"][-1] for feature in features]
			gender_cls_labels = [feature["labels"][-2] for feature in features]
			emotion_cls_labels = [feature["labels"][-3] for feature in features]
			dialect_cls_labels = [feature["labels"][-4] for feature in features]
			#age_cls_labels = [feature["labels"][-1] for feature in features]
			#gender_cls_labels = [feature["labels"][-2] for feature in features]
			#emotion_cls_labels = [feature["labels"][-3] for feature in features]
			#dialect_cls_labels = [feature["labels"][-4] for feature in features]
			#print("Collator age labels: ", age_cls_labels, " gender labels:", gender_cls_labels, "emotion labels:", emotion_cls_labels, "dialect labels:", dialect_cls_labels)
			
		batch = self.processor.pad(
			input_features,
			padding=self.padding,
			max_length=self.max_length,
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors="pt",
		)
		if self.audio_only is False:
			with self.processor.as_target_processor():
				labels_batch = self.processor.pad(
					label_features,
					padding=self.padding,
					max_length=self.max_length_labels,
					pad_to_multiple_of=self.pad_to_multiple_of_labels,
					return_tensors="pt",
				)

			# replace padding with -100 to ignore loss correctly
			ctc_labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
			batch["labels"] = (ctc_labels, torch.tensor(age_cls_labels), torch.tensor(gender_cls_labels), torch.tensor(emotion_cls_labels), torch.tensor(dialect_cls_labels))

		#print("Collator batch is: ", batch)
		return batch