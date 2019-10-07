
import os
import json
import re
from nltk.corpus import wordnet as wn
import sys
def main():
    corpus = {}
    with open('corpus_data/preprocessedf_corpus.json') as file:
        corpus = json.loads(file.read().encode('Utf-8'))

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

    import nltk
    from nltk.corpus import stopwords
    from collections import defaultdict
    from collections import Counter

    nltk.download('wordnet')
    from nltk.corpus import wordnet as wn
    def get_lemma(word):
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma

    from nltk import word_tokenize
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

    STOPWORDS = set(stopwords.words('english'))
    with open('corpus_data/rapsw.txt') as infile:
        infile = infile.read()
        infile = infile.split()
        PROFANITY = set(infile)


    corpus = processLyrics(corpus)

    for author,text in corpus.items():
        corpus[author] = clean_text(text, sys.argv[1])

    artist_shingle = defaultdict(list)
    for artist,lyrics in corpus.items():
        #tokens = [w for w in tokens if not w in sw]
        #shingle3 = set([tuple(tokens[i:i+3]) for i in range(len(tokens) - 3 + 1) if len(tokens[i]) < 10])
        #shingle2 = set([tuple(tokens[i:i+2]) for i in range(len(tokens) - 2 + 1) if len(tokens[i]) < 10])
        shingle1 = lyrics
        # set([tokens[i] for i in range(len(tokens) - 1 + 1) if len(tokens[i]) < 4])
        artist_shingle[artist].append(shingle1)
        #artist_shingle[artist].append(shingle2)
        #artist_shingle[artist].append(shingle3)


    from datasketch import MinHashLSHForest, MinHash
    from sklearn.metrics import jaccard_similarity_score


    listlsh = []
    lsh = MinHashLSHForest(num_perm=128)
    for artist,sets in artist_shingle.items():
        a = MinHash(num_perm=128)
        for d in sets[0]:
            a.update(d.encode('utf8'))
        listlsh.append(a)
        lsh.add(artist,a)

    lsh.index()

    m1 = MinHash(num_perm=128)
    g = []
    with open(sys.argv[2]) as g:
        g = g.read()
        g = g.split()
    for d in g:
        m1.update(d.encode('utf8'))

    result = lsh.query(m1, 5)
    print(" (Up to) Top 5 candidates", result)

if __name__ == "__main__":
    main()
