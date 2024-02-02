import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from torch import nn

class Wav2Vec2ForCTCnCLS(Wav2Vec2PreTrainedModel):

	def __init__(self, config, age_cls_len, gender_cls_len, emotion_cls_len, dialect_cls_len, age_cls_weights, gender_cls_weights, emotion_cls_weights, dialect_cls_weights, alpha=0.01):
		super().__init__(config)
		self.wav2vec2 = Wav2Vec2Model(config)
		self.dropout = nn.Dropout(config.final_dropout)
		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
		self.age_cls_head = nn.Linear(config.hidden_size, age_cls_len)
		self.gender_cls_head = nn.Linear(config.hidden_size, gender_cls_len)
		self.emotion_cls_head = nn.Linear(config.hidden_size, emotion_cls_len)
		self.dialect_cls_head = nn.Linear(config.hidden_size, dialect_cls_len)
		self.init_weights()
		self.age_cls_weights = age_cls_weights
		self.gender_cls_weights = gender_cls_weights
		self.emotion_cls_weights = emotion_cls_weights
		self.dialect_cls_weights = dialect_cls_weights
		self.alpha = alpha

	def freeze_feature_extractor(self):
		self.wav2vec2.feature_extractor._freeze_parameters()

	def _ctc_loss(self, logits, labels, input_values, attention_mask=None):
		loss = None
		if labels is not None:

			# retrieve loss input_lengths from attention_mask
			attention_mask = (
				attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
			)
			input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))

			# assuming that padded tokens are filled with -100
			# when not being attended to
			labels_mask = labels >= 0
			target_lengths = labels_mask.sum(-1)
			flattened_targets = labels.masked_select(labels_mask)

			log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

			with torch.backends.cudnn.flags(enabled=False):
				loss = F.ctc_loss(
					log_probs,
					flattened_targets,
					input_lengths,
					target_lengths,
					blank=self.config.pad_token_id,
					reduction=self.config.ctc_loss_reduction,
					zero_infinity=self.config.ctc_zero_infinity,
					)

		return loss

	# use this function for all classification tasks
	def _cls_loss(self, logits, cls_labels, cls_weights): # sum hidden_states over dim 1 (the sequence length), then feed into self.cls
		loss = None
		if cls_labels is not None:
			loss = F.cross_entropy(logits, cls_labels.to(logits.device), weight=torch.tensor(cls_weights, device=logits.device, dtype=torch.float))
		return loss


	def forward(
		self,
		input_values,
		attention_mask=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
		labels=None, # tuple: (ctc_labels, age_cls_labels, gender_cls_labels), shape=(batch_size, target_length)
		):

		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		outputs = self.wav2vec2(
			input_values,
			attention_mask=attention_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)

		hidden_states = outputs[0] # this is the last layer's hidden states
		hidden_states = self.dropout(hidden_states)

		logits_ctc = self.lm_head(hidden_states)
		logits_age_cls = self.age_cls_head(torch.mean(hidden_states, dim=1))
		logits_gender_cls = self.gender_cls_head(torch.mean(hidden_states, dim=1))
		logits_emotion_cls = self.emotion_cls_head(torch.mean(hidden_states, dim=1))
		logits_dialect_cls = self.dialect_cls_head(torch.mean(hidden_states, dim=1))
		
		loss = None
		if labels is not None:
			#print("labels in forward:", "label1 (age):", labels[1], "label2 (gender):", labels[2], "label3 (emotion):", labels[3], "label4 (dialect):", labels[4])
			
			#print("logits gender:", logits_gender_cls, "gender weights:", self.gender_cls_weights)
			loss_ctc = self._ctc_loss(logits_ctc, labels[0], input_values, attention_mask)
			loss_age_cls = self._cls_loss(logits_age_cls, labels[1], self.age_cls_weights)
			loss_gender_cls = self._cls_loss(logits_gender_cls, labels[2], self.gender_cls_weights)				
			loss_emotion_cls = self._cls_loss(logits_emotion_cls, labels[3], self.emotion_cls_weights)
			loss_dialect_cls = self._cls_loss(logits_dialect_cls, labels[4], self.dialect_cls_weights)
			loss = loss_age_cls + loss_gender_cls + loss_emotion_cls + loss_dialect_cls + self.alpha * loss_ctc

		return CausalLMOutput(
			loss=loss, logits=(logits_ctc, logits_age_cls, logits_gender_cls, logits_emotion_cls, logits_dialect_cls), hidden_states=outputs.hidden_states, attentions=outputs.attentions
		)