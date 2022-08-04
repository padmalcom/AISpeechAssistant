import spacy
# python -m spacy download de_core_news_md
nlp = spacy.load('de_core_news_md')
doc = nlp("Der Affe suchte ein schattiges Plätzchen, um seinen Keks zu essen.")
#doc = nlp("Ich backe leckere Sachen wie Plätzchen oder einen schokoladigen Keks.")
plaetzchen = doc[5]
keks = doc[9]
#print("Spacy - Plätchen-Embedding: ", plaetzchen.vector)
p_sim = plaetzchen.similarity(keks)
print("Spacy - Ähnlichkeit der Begriffe Plätzchen und Keks: ", p_sim)

from gensim.models import Word2Vec, KeyedVectors
# Model von https://devmount.github.io/GermanWordEmbeddings/ herunterladen
model = KeyedVectors.load_word2vec_format("german.model", binary=True)
auto = model['Auto']
#print("Gensim word2vec - Auto-Embedding: ", auto)
auto_synonyms = model.most_similar('Auto', topn=10)
print("Synonyme zu 'Auto' sind: ", auto_synonyms)
