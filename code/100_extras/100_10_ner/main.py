import spacy
from spacy import displacy

nlp = spacy.load('de_core_news_sm')
doc = nlp("Wann hat Barack Obama Geburtstag?")

print(nlp.get_pipe('ner').labels)

for ent in doc.ents:
	print(ent.text, ent.label_)
	
displacy.serve(doc, style="ent")