# Read in from the JSON
import numpy as np
import pandas as pd
import re
import os
import json
import nltk
from collections import defaultdict
import sys
corpus = {}
with open('trainer.json') as corpus:
    corpus = json.loads(corpus.read().encode('utf-8'))

corpus_2 = defaultdict(str)
for artist,songlist in corpus.items():
    for song in songlist:
        lyrics = song['lyrics'].strip('\\')
        corpus_2[artist] += lyrics
features = {}
with open('artist_features.json') as features:
    features = json.loads(features.read())

finalcorpus = []

for artist,lyrics in corpus_2.items():
    d = {}
    d['artist'] = artist
    d['lyrics'] = lyrics
    d['pos'] = features[artist]['pos_counts']
    finalcorpus.append(d)

df = pd.DataFrame(finalcorpus)

# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
"""TOPIC MODELING HOPEFULLY"""
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
PROFANITY = set()
with open('rapsw.txt') as infile:
    infile = infile.read()
    infile = infile.split()
    for el in infile:
        PROFANITY.add(el)

def clean_text(text,ar):
    tokenized_text = word_tokenize(text.lower())
    tokenized_text = [token for token in tokenized_text if len(token) > 5]
    cleaned_text = [t for t in tokenized_text if re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    if ar == 'sw':
        cleaned_text = [t for t in cleaned_text if t not in STOPWORDS]
    if ar == 'lm':
        cleaned_text = [get_lemma(token) for token in cleaned_text]
    if ar =='rw':
        cleaned_text = [token for token in cleaned_text if token not in PROFANITY]
    return cleaned_text
    
for index, row in df.iterrows():
    row['lyrics'] = clean_text(row['lyrics'],sys.argv[2])
from gensim import models, corpora
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import common_corpus, common_texts, get_tmpfile


all_lyrics = []
all_artists = []
for index, row in df.iterrows():
    all_lyrics.append(row['lyrics'])
    all_artists.append(row['artist'])

#common_dictionary = Dictionary(common_texts)
#common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
#lda_model = models.LdaModel(common_corpus, num_topics=10)
dictionary = corpora.Dictionary(all_lyrics)
corpus = [dictionary.doc2bow(text) for text in all_lyrics]

NUM_TOPICS = 25
if sys.argv[1] == 'LDA':
    lda_model = models.LdaModel(corpus=corpus, num_topics=25, id2word=dictionary, passes=20)

    topics = lda_model.print_topics(num_words=4)
    for topic in topics:
        print(topic)
if sys.argv[1] == 'LSI':
    lda_model = models.LsiModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)
    topics = lda_model.print_topics(num_words=4)
    for topic in topics:
        print(topic)

from gensim import similarities


with open('tester.json') as c:
    corpus3 = json.loads(c.read().encode('utf-8'))
total = 0
total1 = 0
total5 = 0
total10 = 0
lda_index = similarities.MatrixSimilarity(lda_model[corpus])

for artist,songlist in corpus3.items():
    for song in songlist:
        bow = dictionary.doc2bow(clean_text(song['lyrics'],sys.argv[2]))
        # Let's perform some queries
        similarities = lda_index[lda_model[bow]]
        # Sort the similarities
        similarities = sorted(enumerate(similarities), key=lambda item: -item[1])
        rl = [all_artists[el[0]] for el in similarities[:10]]
        total += 1
        if artist == rl[0]:
            total1 += 1
        if artist in rl[:5]:
            total5 += 1
        if artist in rl:
            total10 += 1

print('recall@1, @5, @10, total queries')     
print(total1/total)
print(total5/total)
print(total10/total)
print(total)


