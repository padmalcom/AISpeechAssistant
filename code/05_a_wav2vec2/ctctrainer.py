from transformers import Trainer
import torch

class CTCTrainer(Trainer):

	# push input data to (cuda) device
	def _prepare_inputs(self, inputs):
		for k, v in inputs.items():
			if isinstance(v, torch.Tensor):
				kwargs = dict(device=self.args.device)
				inputs[k] = v.to(**kwargs)

			if k == 'labels': # labels are list of tensor, not tensor, special handle here
				new_labels = []
				for i in range(len(inputs[k])):
					new_labels.append(inputs[k][i].to(**kwargs))
				inputs[k] = tuple(new_labels)
				
		if self.args.past_index >= 0 and self._past is not None:
			inputs["mems"] = self._past

		return inputs				
				
	def training_step(self, model, inputs):
		model.train()
		inputs = self._prepare_inputs(inputs)

		loss = self.compute_loss(model, inputs)		

		if self.args.n_gpu > 1:
			loss = loss.mean()

		if self.args.gradient_accumulation_steps > 1:
			loss = loss / self.args.gradient_accumulation_steps

		loss.backward()

		return loss.detach()