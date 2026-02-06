from transformers import MarianMTModel,MarianTokenizer

print("============Normal Translation===========")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

sentences = [
    'How are you?',
    'Hello?',
    'Now ball is in your court'

]

batch = tok(sentences,return_tensors='pt',padding=True)
translated = model.generate(**batch)

for s,t in zip(sentences,translated):
    print(f"{s} -> {tok.decode(t,skip_special_tokens=True)}")

#-----------------reverse translation----------------------
print("========Reverse Translation======")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

sentences = [
    'How are you?',
    'Hello?',
    'Now ball is in your court'

]

batch = tok(sentences,return_tensors='pt',padding=True)
translated = model.generate(**batch)

print("\nTransating english to German!!")
de=[]
for s,t in zip(sentences,translated):
    print(f"{s} -> {tok.decode(t,skip_special_tokens=True)}")
    de.append(tok.decode(t, skip_special_tokens=True))


model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en")
tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

sentences = de

batch = tok(sentences,return_tensors='pt',padding=True)
translated = model.generate(**batch)

print("\nTranslating German To English")
for s,t in zip(sentences,translated):
    print(f"{s} -> {tok.decode(t,skip_special_tokens=True)}")    