from transformers import pipeline
text2text_generator = pipeline(
  "text-generation",
  model="dbmdz/german-gpt2",
  tokenizer="dbmdz/german-gpt2"
)
print(text2text_generator("Wo steht der Triumfbogen? Er steht in "))