""" SVM + MNB  -- Supervised Learning """

import json
import os
import re
# read in the file to dict (from json)

import sys
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

def main():
    corpus = {}
    with open('corpus_data/preprocessedf_corpus.json') as corpus:
        corpus = json.loads(corpus.read().encode('latin-1'))

    finalcorpus = []
    import re
    regex = re.compile('[^a-zA-Z]')
    for artist,songlist in corpus.items():
        for song in songlist:
            d = {}
            d['artist'] = artist
            d['lyrics'] = regex.sub(' ', song['lyrics'])
            d['pos'] = song['pos_counts']
            finalcorpus.append(d)

    df = pd.DataFrame(finalcorpus)
    #print(df.head())
    # read in the file above

    #nltk.download('wordnet')

    from nltk.corpus import wordnet as wn

    import re
    from nltk import word_tokenize
    from nltk.corpus import stopwords
    STOPWORDS = {}
    #stopwords.words('english')
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
    def get_lemma(word):
        lemma = wn.morphy(word)
        if lemma is None:
            return word
        else:
            return lemma

    for index, row in df.iterrows():
        row['lyrics'] = clean_text(row['lyrics'],sys.argv[1])
        lyrics= " ".join(str(x) for x in row['lyrics'])
        poss = ""
        if sys.argv[2] == 'pos':
            for pos,ct in row['pos'].items():
                poss += ((pos+" ") * (int(int(ct)/40)))
            lyrics = lyrics + poss
        df.loc[index,'lyric_pos'] = lyrics

    np.random.seed(500)

    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['lyric_pos'],df['artist'],test_size=0.3)

    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer


    Tfidf_vect = TfidfVectorizer()
    Tfidf_vect.fit(df['lyric_pos'])


    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    Naive = naive_bayes.MultinomialNB()
    Naive.fit(Train_X_Tfidf,Train_Y)
    # predict the labels on validation dataset

    predictions_NB = Naive.predict(Test_X_Tfidf)
    # Use accuracy_score function to get the accuracy
    print('TFIDF Based Models')
    print("TFIDF Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y))


    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf,Train_Y)
    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    # Use accuracy_score function to get the accuracy
    print("TFIDF SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y))

    print('')

    print('COUNT BASED Models')
    Count_vect = CountVectorizer()
    Count_vect.fit(df['lyric_pos'])


    Train_X_ct = Count_vect.transform(Train_X)
    Test_X_ct = Count_vect.transform(Test_X)

    Naive = naive_bayes.MultinomialNB()
    Naive.fit(Train_X_ct,Train_Y)
    # predict the labels on validation dataset

    predictions_NB = Naive.predict(Test_X_ct)
    # Use accuracy_score function to get the accuracy
    print("CT Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y))


    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf,Train_Y)
    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_ct)
    # Use accuracy_score function to get the accuracy
    print("CT SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y))

if __name__ == "__main__":
    main()
