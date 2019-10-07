import os
import json
from collections import defaultdict,Counter
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

corpus = {}
with open('corpus_data/preprocessed_corpus.json') as corpus:
    corpus = json.loads(corpus.read().encode('utf-8'))

corpus_2 = defaultdict(str)
for artist,songlist in corpus.items():
    for song in songlist:
        lyrics = song['lyrics'].strip('\\')
        corpus_2[artist] += lyrics


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
STOPWORDS = stopwords.words('english')
profanity = set(['fuck','bitch','nigga','shit','money','right','never','fuckin','fucking','never','motherfucker'])
def clean_text(text):
    tokenized_text = word_tokenize(text.lower())
    tokenized_text = [token for token in tokenized_text if len(token) > 4]
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    cleaned_text = [get_lemma(token) for token in cleaned_text]
    #cleaned_text = [token for token in cleaned_text if token not in profanity]
    return cleaned_text
common = Counter()
for artist,lyrics in corpus_2.items():
    common += Counter(clean_text(lyrics))

print(common.most_common(100))

with open('corpus_data/rapsw.txt','w') as ofile:
    for key in common.most_common(100):
        ofile.write(key[0] + " \n")