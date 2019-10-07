import sys
import os
from collections import defaultdict
from text_preprocess import txt_preprocesser
import math
import json
"""KMAMIN 62182275 KRISHAN AMIN"""

class bayesian_classifier:

    def trainNaiveBayes(self,train_list):
        preprocesser = txt_preprocesser()
        class_doc_counts = defaultdict(int)
        class_wc = defaultdict(lambda: defaultdict(int))
        word_counts = set()
        classes = set()
        class_wtotals = defaultdict(int)
        for element in train_list:
            artist = element['artist']
            classes.add(artist)
            word_list = preprocesser.process(element['lyrics'],sys.argv[1])
            class_doc_counts[artist] += 1
            for word in word_list:
                class_wc[artist][word] += 1
                word_counts.add(word)
                class_wtotals[artist] += 1
        return [class_doc_counts,class_wc,word_counts,class_wtotals,classes]

    def testNaiveBayes(self):
        preprocesser = txt_preprocesser() # declare preprocessor
            # lcount = 0
            # tcount = 0 used for accuracy readnigs
            # accuracy = 0
        finalcorpus = []
        with open('trainer.json') as corpus:
            corpus = json.loads(corpus.read().encode('latin-1'))

            for artist,songlist in corpus.items():
                for song in songlist:
                    d = {}
                    d['artist'] = artist
                    d['lyrics'] = song['lyrics']
                    finalcorpus.append(d)
        td = self.trainNaiveBayes(finalcorpus) # TRAIN
        cdc = td[0] # docs in true and lie
        cwc = td[1] # word counts in true | lie
        word_counts = td[2] # word counts overall
        cwt = td[3] # total words in true / false 
        class_list = td[4]
        class_score = defaultdict(int)
        for el in class_list:
            class_score[el] = 0
        # RETURN ALL NECC COUNTS
        numdocs = sum(val for key,val in cdc.items())
        
        tester = {}
        with open('tester.json') as file:
            tester = json.loads(file.read().encode('latin-1'))
        totalnum = 0
        totalcorrect = 0
        accuracy = 0
        total5 = 0
        for aartist,songlist in tester.items():
            for song in songlist:
                for el in class_list:
                    class_score[el] = 0
                wlist = preprocesser.process(song['lyrics'],sys.argv[1])        
                for artist,score in class_score.items():
                    for word in wlist:
                        score += math.log((1 + cwc[artist][word]) / (cwt[artist]))
                    score *= cdc[artist]/numdocs  # add the P(lie)
                    class_score[artist] = score
                sorted_results =  sorted(class_score.items(), key=lambda kv: kv[1], reverse=True)
                sorted_results = sorted_results[:10]
                sorted_results = [e[0] for e in sorted_results]
                if aartist in sorted_results:
                    totalcorrect += 1
                if aartist == sorted_results[0]:
                    accuracy += 1
                if aartist in sorted_results[:5]:
                    total5 += 1
                totalnum += 1
        print('Recall@1,@5,@10 , total')
        print(accuracy/totalnum)
        print(total5/totalnum)
        print(totalcorrect/totalnum)
        print(totalnum)

def main():
    classifier = bayesian_classifier()
    classifier.testNaiveBayes()


if __name__ == "__main__":
    main()
