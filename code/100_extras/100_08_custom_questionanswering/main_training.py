# Inspiration: https://github.com/huggingface/transformers/tree/master/examples/pytorch/question-answering
# Modelle und Trainingsdaten: https://www.deepset.ai/
from loguru import logger
import os

from transformers import (
	AutoConfig,
	AutoModelForQuestionAnswering,
	AutoTokenizer,
	EvalPrediction,
	HfArgumentParser,
	TrainingArguments,
	default_data_collator,
	set_seed
)

from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, load_metric

from preprocess import prepare_train_features, prepare_validation_features
from postprocess import post_processing_function
from qatrainer import QuestionAnsweringTrainer
from arguments import ModelArguments, DataTrainingArguments

# Deaktiviere Weights&Biases (WandB) Online Service (deprecated, aber funktioniert)
os.environ["WANDB_DISABLED"] = "true"

training_args = {
	"n_gpu":1,
	"model_name_or_path":"bert-base-german-cased",
	"dataset_name":"deepset/germanquad",
	"max_seq_length":384,
	"output_dir":"./models",
	"per_device_train_batch_size":12,
	"per_device_eval_batch_size":12, 
	"learning_rate":3e-05,
	"num_train_epochs":10,
	"doc_stride":128,
	"save_steps":5000,
	"logging_steps":5000,
	"seed":42 
}

if __name__ == '__main__':

	logger.info("Parse Parameter...")
	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
	model_args, data_args, training_args = parser.parse_dict(training_args)
	
	# Setze festen Start für Zufallswerte von NumPy, Torch etc. -> Reproduzierbarkeit
	set_seed(training_args.seed)
	
	logger.info("Lade Trainings- und Testdaten...")
	datasets = load_dataset(data_args.dataset_name, None)		
	logger.info("Aufbau des Datensatzes: {}.", datasets)
	
	logger.info("Erstelle Konfiguration, Tokenizer und Modell für QA...")
	config = AutoConfig.from_pretrained(
		model_args.model_name_or_path
	)
	
	tokenizer = AutoTokenizer.from_pretrained(
		model_args.model_name_or_path
	)
	
	model = AutoModelForQuestionAnswering.from_pretrained(
		model_args.model_name_or_path,
		from_tf=bool(".ckpt" in model_args.model_name_or_path),
		config=config
	)
	
	logger.info("Beispielhafter Datensatz: {}", datasets["train"][0])
	logger.info("Beispiel Tokenisierung: {}", tokenizer(datasets["train"][10]["context"]))
	
	# Auswahl der Spalten für Frage, Kontext und Antwort
	column_names = datasets["train"].column_names
	question_column_name = "question" if "question" in column_names else column_names[0]
	context_column_name = "context" if "context" in column_names else column_names[1]
	answer_column_name = "answers" if "answers" in column_names else column_names[2]
	
	logger.info("Spalte für Fragen '{}', Kontexte '{}' und Antworten '{}' gesetzt.", question_column_name, context_column_name, answer_column_name)
	
	logger.info("Bereite Trainingsdaten vor...")
	train_dataset = datasets["train"].map(
		prepare_train_features,
		batched=True,
		remove_columns=column_names,
		fn_kwargs=dict(tokenizer=tokenizer, question_column_name=question_column_name, context_column_name=context_column_name, answer_column_name=answer_column_name, 
			max_seq_length=data_args.max_seq_length, doc_stride=data_args.doc_stride)
	)

	
	logger.info("Bereite Testdaten vor...")
	validation_dataset = datasets["test"].map(
		prepare_validation_features,
		batched=True,
		remove_columns=column_names,
		fn_kwargs=dict(tokenizer=tokenizer, question_column_name=question_column_name, context_column_name=context_column_name, max_seq_length=data_args.max_seq_length,
			doc_stride=data_args.doc_stride)
	)

	data_collator = default_data_collator

	logger.info("Lade Metriken für SQuAD2 ...")
	metric = load_metric("squad_v2")
	
	def compute_metrics(p: EvalPrediction):
		return metric.compute(predictions=p.predictions, references=p.label_ids)
	
	logger.info("Initialisiere Training...")
	trainer = QuestionAnsweringTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=validation_dataset,
		eval_examples=datasets["test"],
		tokenizer=tokenizer,
		data_collator=data_collator,
		post_process_function=post_processing_function,
		compute_metrics=compute_metrics,
		
		n_best_size = data_args.n_best_size,
		max_answer_length = data_args.max_answer_length,
		null_score_diff_threshold=data_args.null_score_diff_threshold,
		output_dir=training_args.output_dir,
		answer_column_name=answer_column_name
	)
	
	last_checkpoint = get_last_checkpoint(training_args.output_dir)
	if last_checkpoint is not None:
		checkpoint = last_checkpoint
	elif os.path.isdir(model_args.model_name_or_path):
		checkpoint = model_args.model_name_or_path
	else:
		checkpoint = None
	logger.info("Letzter Trainings-Checkpoint: {}", checkpoint)
	
	logger.info("Starte Training...")
	train_result = trainer.train(resume_from_checkpoint=checkpoint)
	
	# Speichere das Modell nach dem Training
	trainer.save_model()

	logger.info("Speichere Ergebnis...")
	output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
	if trainer.is_world_process_zero():
		with open(output_train_file, "w") as writer:
			logger.info("Ergebnis des Trainings:")
			for key, value in sorted(train_result.metrics.items()):
				logger.info(f"  {key} = {value}")
				writer.write(f"{key} = {value}\n")

		trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

	# Evaluation
	logger.info("Evaluiere Modell...")
	results = {}
	results = trainer.evaluate()

	logger.info("Speichere Evaluationsdaten...")
	output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
	if trainer.is_world_process_zero():
		with open(output_eval_file, "w") as writer:
			logger.info("Evaluationsergebnisse:")
			for key, value in sorted(results.items()):
				logger.info(f"  {key} = {value}")
				writer.write(f"{key} = {value}\n")
	logger.info("Ergebnis: {}", results)	