# Read in from the JSON
import numpy as np
import pandas as pd
import re
import os
import json
import nltk
from collections import defaultdict
import sys

def main():
    corpus = {}
    with open('corpus_data/preprocessedf_corpus.json') as corpus:
        corpus = json.loads(corpus.read().encode('utf-8'))

    corpus_2 = defaultdict(str)
    for artist,songlist in corpus.items():
        for song in songlist:
            lyrics = song['lyrics'].strip('\\')
            corpus_2[artist] += lyrics
    features = {}
    with open('corpus_data/artist_features.json') as features:
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
    with open('corpus_data/rapsw.txt') as infile:
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
        row['lyrics'] = clean_text(row['lyrics'],sys.argv[1])
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
    lda_model = models.LdaModel(corpus=corpus, num_topics=25, id2word=dictionary, passes=20)

    topics = lda_model.print_topics(num_words=4)
    print('LDA Topics')
    for topic in topics:
        print(topic)

    lsi_model = models.LsiModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)
    topics = lsi_model.print_topics(num_words=4)
    print ('LSI TOPICS')
    for topic in topics:
        print(topic)

    from gensim import similarities

    text = ""
    with open(sys.argv[2]) as inf:
        inf = inf.read()
        text = inf

    bow = dictionary.doc2bow(clean_text(text,sys.argv[1]))
    lda_index = similarities.MatrixSimilarity(lda_model[corpus])
    lsi_index = similarities.MatrixSimilarity(lsi_model[corpus])
    # Let's perform some queries
    similarities = lda_index[lda_model[bow]]
    # Sort the similarities
    similarities = sorted(enumerate(similarities), key=lambda item: -item[1])


    similaritiesLSI = lsi_index[lsi_model[bow]]

    similaritiesLSI = sorted(enumerate(similaritiesLSI), key=lambda item: -item[1])

    # Top most similar documents:
    #print(similarities[:10])
    # [(104, 0.87591344), (178, 0.86124849), (31, 0.8604598), (77, 0.84932965), (85, 0.84843522), (135, 0.84421808), (215, 0.84184396), (353, 0.84038532), (254, 0.83498049), (13, 0.82832891)]

    # Let's see what's the most similar document
    document_id, similarity = similarities[0]
    document_id2, similarityLSI = similaritiesLSI[0]

    # print(all_lyrics[document_id][:1000])
    print("LDA : TOP 5 Similar ARTISTS")
    for el in similarities[:5]:
        print(all_artists[el[0]])

    print('')
    print('LSI : Top 5 Similar Artists')
    for el in similaritiesLSI[:5]:
        print(all_artists[el[0]])

if __name__ == "__main__":
    main()
