import nltk
# import spacy
import os
import json
from collections import Counter

corpus = {}
with open('corpus_data/preprocessedf_corpus.json') as infile:
    corpus = json.loads(infile.read().encode('utf-8'))

featurecounts = {}
for artist,songlist in corpus.items():
    features = {"pos_counts":Counter()}
    numwords = 0
    linelen = 0
    charlength = 0
    for song in songlist:
        x = Counter(song['pos_counts'])
        features['pos_counts'] += x
        charlength += sum(len(word) for word in song['lyrics'].split())
        numwords += len(song['lyrics'].split())
        linelen += song['avg_linelen']
    for name,count in features['pos_counts'].items():
        features['pos_counts'][name] /= numwords

    features['avg_wordlen'] = charlength / numwords
    features['avg_linelen'] = linelen / len(songlist)
    featurecounts[artist] = features

with open('corpus_data/artist_features.json','w') as ofile:
    json.dump(featurecounts,ofile,indent=4)



"""
import spacy
songlyrics = nlp(song['lyrics'])
pos_counts = dict(Counter([token.pos_ for token in songlyrics]))
for name,count in pos_counts.items():
    features['pos_counts'][name] /= len(songlyrics)
"""
