# Quelle: https://github.com/explosion/spaCy/blob/v2.3.x/examples/training/train_intent_parser.py
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# Trainingsdaten bestehend aus Text, Token Heads (das Wort auf, das sich dieses Wort bezieht) und syntaktischer Abhängikeit (Toeken Dependency)
# "-" wird eingesetzt, falls es keine Beziehung gibt.
TRAIN_DATA = [

    (
        "finde einen urlaubsort mit viel sonne",
        {
            "heads": [0, 2, 0, 5, 5, 2],
            "deps": ["ROOT", "-", "PLACE", "-", "QUALITY", "ATTRIBUTE"],
        },
    ),
	
    (
        "suche ein hotel nahe einer strandbar",
        {
            "heads": [0, 2, 0, 5, 5, 2],
            "deps": ["ROOT", "-", "PLACE", "QUALITY", "-", "ATTRIBUTE"],
        },
    ),
	
    (
        "finde ein appartment mit meerblick",
        {
            "heads": [0, 2, 0, 4, 2],
            "deps": ["ROOT", "-", "PLACE", "-", "ATTRIBUTE"],
        },
    ),
	
	(
        "suche ein restaurant mit italienischem Essen",
        {
            "heads": [0, 2, 0, 5, 5, 2],
            "deps": ["ROOT", "-", "PLACE", "-", "ATTRIBUTE", "PRODUCT"],
        },
    ),
	
	(
        "finde eine tankstelle die blumen verkauft",
        {
            "heads": [0, 2, 0, 2, 2, 2],
            "deps": ["ROOT", "-", "PLACE", "-", "PRODUCT", "-"],
        },
    ),
	
	(
        "finde ein gutes veganes cafe nahe des parks",
        {
            "heads": [0, 4, 4, 4, 0, 7, 7, 4],
            "deps": ["ROOT", "-", "QUALITY", "ATTRIBUTE", "PLACE", "ATTRIBUTE", "-", "LOCATION"],
        },
    ),
]

def test_model(nlp):
    texts = [
        "finde ein hotel mit gutem internet",
        "suche das günstiges fitnessstudio nahe der arbeit",
        "suche das beste restaurant in berlin",
    ]
    docs = nlp.pipe(texts)
    for doc in docs:
        print(doc.text)
        print([(t.text, t.dep_, t.head.text) for t in doc if t.dep_ != "-"])


if __name__ == "__main__":
	output_dir = None
	iterations=15
    nlp = spacy.blank("de")

    if "parser" in nlp.pipe_names:
        nlp.remove_pipe("parser")
    parser = nlp.create_pipe("parser")
    nlp.add_pipe(parser, first=True)

    for text, annotations in TRAIN_DATA:
        for dep in annotations.get("deps", []):
            parser.add_label(dep)

    pipe_exceptions = ["parser", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, losses=losses)
            print("Losses", losses)

    # test the trained model
    test_model(nlp)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        test_model(nlp2)