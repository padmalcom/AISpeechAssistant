import spacy
from spacy import displacy

nlp = spacy.load("de_core_news_sm")
doc = nlp("Goldfische sind pflegeleicht.")
print(spacy.explain('sb'))
print(spacy.explain('pd'))
displacy.serve(doc, style="dep")
