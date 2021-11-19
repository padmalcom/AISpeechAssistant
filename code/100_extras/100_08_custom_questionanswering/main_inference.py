import torch
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer
)

tokenizer = AutoTokenizer.from_pretrained("./models/checkpoint-20000")
config = AutoConfig.from_pretrained("./models/checkpoint-20000")
model = AutoModelForQuestionAnswering.from_pretrained("./models/checkpoint-20000", config=config)

text = "i live in berlin."
question = "where do i live?"
enforce_answer = True

encoding = tokenizer.encode_plus(question, text, return_tensors="pt")
input_ids = encoding["input_ids"]

# default is local attention everywhere
# the forward method will automatically set global attention on question tokens
attention_mask = encoding["attention_mask"]
model.cuda()

with torch.no_grad():
  logits = model(input_ids.cuda(), attention_mask.cuda())
  
all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
print(all_tokens)

st = 1 if enforce_answer else 0
print(torch.argmax(logits.start_logits[0][st:]))
print(torch.argmax(logits.end_logits[0][st:]))

answer_tokens = all_tokens[torch.argmax(logits.start_logits[0][st:]) :torch.argmax(logits.end_logits[0][st:])+1]
answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))

print("Predicted Answer:", answer)