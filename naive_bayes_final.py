import sys
import os
from collections import defaultdict
from text_preprocess import txt_preprocesser
import math
import json
"""KMAMIN 62182275 KRISHAN AMIN"""

class bayesian_classifier:
    def trainNaiveBayes(self,addition):
        trainlist = []
        with open('corpus_data/preprocessedf_corpus.json') as corpus:
            corpus = json.loads(corpus.read().encode('latin-1'))

            for artist,songlist in corpus.items():
                for song in songlist:
                    d = {}
                    d['artist'] = artist
                    d['lyrics'] = song['lyrics']
                    trainlist.append(d)
        preprocesser = txt_preprocesser()
        class_doc_counts = defaultdict(int)
        class_wc = defaultdict(lambda: defaultdict(int))
        word_counts = {}
        classes = {}
        class_wtotals = defaultdict(int)
        for element in trainlist:
            artist = element['artist']
            classes[(artist)] = 1
            word_list = preprocesser.process(element['lyrics'],addition)
            class_doc_counts[artist] += 1
            for word in word_list:
                class_wc[artist][word] += 1
                word_counts[word] = 1
                class_wtotals[artist] += 1
        outputdict = {"cdc":class_doc_counts,'cwc':class_wc,'wc':word_counts,'cwt':class_wtotals,'clist':classes}
        """with open('nb_train.json','w') as ofile:
            json.dump(outputdict,ofile,indent=4)
            ofile.close()"""
        return outputdict

    def testNaiveBayes(self,addition):
        preprocesser = txt_preprocesser() # declare preprocessor
            # lcount = 0
            # tcount = 0 used for accuracy readnigs
            # accuracy = 0
        td = self.trainNaiveBayes(addition[0])
        """with open('nb_train.json',) as ifile:
            td = json.loads(ifile.read())"""
         # TRAIN
        cdc = td['cdc'] # docs in true and lie
        cwc = td['cwc'] # word counts in true | lie
        word_counts = td['wc'] # word counts overall
        cwt = td['cwt'] # total words in true / false
        class_list = td['clist']
        class_score = defaultdict(int)
        for el in class_list:
            class_score[el] = 0
        # RETURN ALL NECC COUNTS
        numdocs = sum(val for key,val in cdc.items())
        with open(addition[1]) as file:
            wlist = preprocesser.process(file.read(),addition)
        for artist,score in class_score.items():
            for word in wlist:
                score += math.log((1+ cwc[artist].get(word,0)) / (cwt[artist]))

                # lie score += log( p(lie) * count(word | lie) / (vocabsize + vocabsize of lies))
            score *= cdc[artist]/numdocs  # add the P(lie)
            class_score[artist] = score
        sorted_results =  sorted(class_score.items(), key=lambda kv: kv[1], reverse=True)
        for key, value in sorted_results[:5]:
            print(str(key) + "\t" + str(value))


def main():
    classifier = bayesian_classifier()
    #classifier.trainNaiveBayes()
    classifier.testNaiveBayes(sys.argv[1:])


if __name__ == "__main__":
    main()
