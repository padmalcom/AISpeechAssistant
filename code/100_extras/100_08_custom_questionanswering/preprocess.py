# Training preprocessing
def prepare_train_features(examples, tokenizer, question_column_name, context_column_name, answer_column_name, max_seq_length, doc_stride):
	pad_on_right = tokenizer.padding_side == "right"
	
	# Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
	# in one example possible giving several features when a context is long, each of those features having a
	# context that overlaps a bit the context of the previous feature.
	tokenized_examples = tokenizer(
		examples[question_column_name if pad_on_right else context_column_name],
		examples[context_column_name if pad_on_right else question_column_name],
		truncation="only_second" if pad_on_right else "only_first",
		max_length=max_seq_length,
		stride=doc_stride,
		return_overflowing_tokens=True,
		return_offsets_mapping=True,
		padding="max_length"
	)

	# Since one example might give us several features if it has a long context, we need a map from a feature to
	# its corresponding example. This key gives us just that.
	sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
	# The offset mappings will give us a map from token to character position in the original context. This will
	# help us compute the start_positions and end_positions.
	offset_mapping = tokenized_examples.pop("offset_mapping")

	# Let's label those examples!
	tokenized_examples["start_positions"] = []
	tokenized_examples["end_positions"] = []
	tokenized_examples["is_impossible"] = [] 
	tokenized_examples["cls_index"] = [] 

	for i, offsets in enumerate(offset_mapping):
		# We will label impossible answers with the index of the CLS token.
		input_ids = tokenized_examples["input_ids"][i]
		cls_index = input_ids.index(tokenizer.cls_token_id)
		tokenized_examples["cls_index"].append(cls_index) 

		# Grab the sequence corresponding to that example (to know what is the context and what is the question).
		sequence_ids = tokenized_examples.sequence_ids(i)

		# One example can give several spans, this is the index of the example containing this span of text.
		sample_index = sample_mapping[i]
		answers = examples[answer_column_name][sample_index]
		# If no answers are given, set the cls_index as answer.
		if len(answers["answer_start"]) == 0:
			tokenized_examples["start_positions"].append(cls_index)
			tokenized_examples["end_positions"].append(cls_index)
			tokenized_examples["is_impossible"].append(1.0) 
		else:
			# Start/end character index of the answer in the text.
			start_char = answers["answer_start"][0]
			end_char = start_char + len(answers["text"][0])

			# Start token index of the current span in the text.
			token_start_index = 0
			while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
				token_start_index += 1

			# End token index of the current span in the text.
			token_end_index = len(input_ids) - 1
			while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
				token_end_index -= 1

			# Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
			if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
				tokenized_examples["start_positions"].append(cls_index)
				tokenized_examples["end_positions"].append(cls_index)
				tokenized_examples["is_impossible"].append(1.0) 
			else:
				# Otherwise move the token_start_index and token_end_index to the two ends of the answer.
				# Note: we could go after the last offset if the answer is the last word (edge case).
				while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
					token_start_index += 1
				tokenized_examples["start_positions"].append(token_start_index - 1)
				while offsets[token_end_index][1] >= end_char:
					token_end_index -= 1
				tokenized_examples["end_positions"].append(token_end_index + 1)
				tokenized_examples["is_impossible"].append(0.0)
				
	return tokenized_examples
	
def prepare_validation_features(examples, tokenizer, question_column_name, context_column_name, max_seq_length, doc_stride):

	pad_on_right = tokenizer.padding_side == "right"
	# Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
	# in one example possible giving several features when a context is long, each of those features having a
	# context that overlaps a bit the context of the previous feature.
	tokenized_examples = tokenizer(
		examples[question_column_name if pad_on_right else context_column_name],
		examples[context_column_name if pad_on_right else question_column_name],
		truncation="only_second" if pad_on_right else "only_first",
		max_length=max_seq_length,
		stride=doc_stride,
		return_overflowing_tokens=True,
		return_offsets_mapping=True,
		padding="max_length"
	)

	# Since one example might give us several features if it has a long context, we need a map from a feature to
	# its corresponding example. This key gives us just that.
	sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

	# For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
	# corresponding example_id and we will store the offset mappings.
	tokenized_examples["example_id"] = []

	# We still provide the index of the CLS token, but not the is_impossible label. 
	tokenized_examples["cls_index"] = [] 

	for i in range(len(tokenized_examples["input_ids"])):
		# Find the CLS token in the input ids. 
		input_ids = tokenized_examples["input_ids"][i]
		cls_index = input_ids.index(tokenizer.cls_token_id)
		tokenized_examples["cls_index"].append(cls_index) 

		# Grab the sequence corresponding to that example (to know what is the context and what is the question).
		sequence_ids = tokenized_examples.sequence_ids(i)
		context_index = 1 if pad_on_right else 0

		# One example can give several spans, this is the index of the example containing this span of text.
		sample_index = sample_mapping[i]
		tokenized_examples["example_id"].append(examples["id"][sample_index])

		# Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
		# position is part of the context or not.
		tokenized_examples["offset_mapping"][i] = [
			(o if sequence_ids[k] == context_index else None)
			for k, o in enumerate(tokenized_examples["offset_mapping"][i])
		]

	return tokenized_examples
	