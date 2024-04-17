import os
import json
from transformers import Wav2Vec2CTCTokenizer

def build_tokenizer(model_output_dir, dataset):

	def extract_all_chars(batch):
		all_text = " ".join(batch["sentence"]).replace("<unk>", "")
		return {"all_text": [all_text]}

	vocab_train = dataset["train"].map(
		extract_all_chars,
		batched=True,
		batch_size=-1,
		remove_columns=dataset.column_names["train"],
		num_proc=1
	)
	
	special_vocab_dict = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4}

    vocab_list = set(vocab_train["all_text"][0])

    # Behalte alle Buchstaben (a-zA-Z), Bindestriche und Anführungszeichen
	vocab_list = [x for x in vocab_list if x.isalpha() or x in ["-", "\""]]

	vocab_list = sorted(vocab_list)
	vocab_dict = {v: k + len(special_vocab_dict) for k, v in enumerate(vocab_list)}
	vocab_dict = dict(special_vocab_dict, **vocab_dict)

	vocab_path = os.path.join(model_output_dir, "vocab.json")

	with open(vocab_path, "w") as vocab_file:
		json.dump(vocab_dict, vocab_file)

	return Wav2Vec2CTCTokenizer(
		vocab_path,
		unk_token="<unk>",
		pad_token="<pad>",
		word_delimiter_token="|",
	)