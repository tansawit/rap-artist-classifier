
import os
import json
import re
from nltk.corpus import wordnet as wn
import sys 
corpus = {}
with open('trainer.json') as file:
    corpus = json.loads(file.read().encode('utf-8'))

from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
import nltk
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
PROFANITY = set()
with open('rapsw.txt') as infile:
    infile = infile.read()
    infile = infile.split()
    for el in infile:
        PROFANITY.add(el)

def clean_text(text):
    tokenized_text = word_tokenize(text.lower())
    tokenized_text = [token for token in tokenized_text if len(token) > 5]
    cleaned_text = [t for t in tokenized_text if re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    if sys.argv[1] == 'sw':
        cleaned_text = [t for t in cleaned_text if t not in STOPWORDS]
    if sys.argv[1] == 'lm':
        cleaned_text = [get_lemma(token) for token in cleaned_text]
    if sys.argv[1] =='rw':
        cleaned_text = [token for token in cleaned_text if token not in PROFANITY]
    return cleaned_text
def processLyrics(lyrics):
    authors = {}
    for author in lyrics:
        for song in lyrics[author]:
            lyric = re.sub(r'\[[^>]+\]', '', song["lyrics"])
            lyric = re.sub(r'\([^>]+\)', '', lyric)
            lyric = re.sub(r'\{[^>]+\}', '', lyric)
            lyric = lyric.split(r'\s')
            for line in lyric:
                line = re.sub(r'\n', ' ', line)
                if author not in authors:
                    authors[author] = line
                else:
                    authors[author] += line
    return authors
from nltk.corpus import stopwords
from collections import defaultdict
from collections import Counter
sw = set(stopwords.words('english'))
artist_shingle = defaultdict(list)
corpus = processLyrics(corpus)
for artist,lyrics in corpus.items():
    tokens = clean_text(lyrics)
    artist_shingle[artist].append(tokens)


from datasketch import MinHashLSHForest, MinHash
from sklearn.metrics import jaccard_similarity_score

g = []

listlsh = []
lsh = MinHashLSHForest(num_perm=128)
for artist,sets in artist_shingle.items():
    a = MinHash(num_perm=128)
    for d in sets[0]:
        a.update(d.encode('utf8'))
    listlsh.append(a)
    lsh.add(artist,a)

lsh.index()
tester = {}
with open('tester.json') as file:
    tester = json.loads(file.read().encode('latin-1'))
numcorrect_1 =0
numcorrect_5 = 0
numcorrect_10 = 0
total = 0
for artist,songlist in tester.items():
    for song in songlist:
        m1 = MinHash(num_perm=128)
        songp = clean_text(song['lyrics'])
        for d in songp:
            m1.update(d.encode('utf8'))
        result = lsh.query(m1, 10)
        if len(result):
            total += 1
            if artist in result:
                numcorrect_10 += 1
            if len(result) >= 5:
                if artist in result[:5]:
                    numcorrect_5 += 1
            if artist == result[0]:
                numcorrect_5 += 1
print("Recall @1,@5,@10, total")
print(numcorrect_1/total)
print(numcorrect_5/total)
print(numcorrect_10/total)
print(total)