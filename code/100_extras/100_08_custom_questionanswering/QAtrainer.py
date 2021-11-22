from transformers import Trainer, is_datasets_available, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput


if is_datasets_available():
	import datasets

if is_torch_tpu_available():
	import torch_xla.core.xla_model as xm
	import torch_xla.debug.metrics as met


class QuestionAnsweringTrainer(Trainer):
	def __init__(self, *args, eval_examples=None, post_process_function=None, n_best_size, max_answer_length, null_score_diff_threshold, output_dir, answer_column_name, **kwargs):
		super().__init__(*args, **kwargs)
		self.eval_examples = eval_examples
		self.post_process_function = post_process_function
		self.n_best_size = n_best_size
		self.max_answer_length = max_answer_length
		self.null_score_diff_threshold = null_score_diff_threshold
		self.output_dir = output_dir
		self.answer_column_name = answer_column_name

	def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
		eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
		eval_dataloader = self.get_eval_dataloader(eval_dataset)
		eval_examples = self.eval_examples if eval_examples is None else eval_examples

		# Temporarily disable metric computation, we will do it in the loop here.
		compute_metrics = self.compute_metrics
		self.compute_metrics = None
		print("Data loader: " + str(eval_dataloader))
		try:
			output = self.prediction_loop(
				eval_dataloader,
				description="Evaluation",
				# No point gathering the predictions if there are no metrics, otherwise we defer to
				# self.args.prediction_loss_only
				prediction_loss_only=True if compute_metrics is None else None,
				ignore_keys=ignore_keys,
			)
			print(output)
		finally:
			self.compute_metrics = compute_metrics

		# We might have removed columns from the dataset so we put them back.
		if isinstance(eval_dataset, datasets.Dataset):
			eval_dataset.set_format(type=eval_dataset.format["type"], columns=list(eval_dataset.features.keys()))

		if self.post_process_function is not None and self.compute_metrics is not None:
			eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions, self.n_best_size, self.max_answer_length, self.null_score_diff_threshold, self.output_dir, self.answer_column_name)
			metrics = self.compute_metrics(eval_preds)

			self.log(metrics)
		else:
			metrics = {}

		if self.args.tpu_metrics_debug or self.args.debug:
			# tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
			xm.master_print(met.metrics_report())

		self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
		return metrics

	def predict(self, test_dataset, test_examples, ignore_keys=None):
		test_dataloader = self.get_test_dataloader(test_dataset)

		# Temporarily disable metric computation, we will do it in the loop here.
		compute_metrics = self.compute_metrics
		self.compute_metrics = None
		try:
			output = self.prediction_loop(
				test_dataloader,
				description="Evaluation",
				# No point gathering the predictions if there are no metrics, otherwise we defer to
				# self.args.prediction_loss_only
				prediction_loss_only=True if compute_metrics is None else None,
				ignore_keys=ignore_keys,
			)
		finally:
			self.compute_metrics = compute_metrics

		if self.post_process_function is None or self.compute_metrics is None:
			return output

		# We might have removed columns from the dataset so we put them back.
		if isinstance(test_dataset, datasets.Dataset):
			test_dataset.set_format(type=test_dataset.format["type"], columns=list(test_dataset.features.keys()))

		eval_preds = self.post_process_function(test_examples, test_dataset, output.predictions)
		metrics = self.compute_metrics(eval_preds)

		return PredictionOutput(predictions=eval_preds.predictions, label_ids=eval_preds.label_ids, metrics=metrics)