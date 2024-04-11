from transformers import Trainer
import torch

class CTCTrainer(Trainer):

	# push input data to (cuda) device
	def _prepare_inputs2(self, inputs):
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
				
    # Überschreiben der Methode training_step der Klasse Trainer
	def training_step(self, model, inputs):
    
        # Setze das Modell in en Trainingsmodus
		model.train()
        
        # Bereite die Eingabedaten für das Modell vor
		inputs = self._prepare_inputs(inputs)

        # Berechne das Loss für die Eingabedaten
		loss = self.compute_loss(model, inputs)		

        # Sind mehrere GPUs verfügbar, berechne das mittlere Loss
		if self.args.n_gpu > 1:
			loss = loss.mean()

        # Wird Gradientenakkumulation eingesetzt, wird das Loss entsprechend skaliert
		if self.args.gradient_accumulation_steps > 1:
			loss = loss / self.args.gradient_accumulation_steps

        # Gebe das loss durch Backpropagation zurück, sodass die Gradienten angepasst werden können
		loss.backward()

        # Gib das Loss, losgelöst vom aktuellen Graph, zurück
		return loss.detach()