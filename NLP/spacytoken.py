#python -m spacy download en_core_web_sm

import spacy
nlp = spacy.load("en_core_web_sm")

text = "Apple's runner was running quickly towards the finish line, but he wasn't sure why."
doc = nlp(text)
for token in doc:
    print(token.text, "→", token.lemma_, token.pos_, "stop?" , token.is_stop)
