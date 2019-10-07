import os
import re
import json
import spacy
import codecs
from collections import Counter
def main():
    lyrics = {}
    with open('corpus_data/newlyrics.json') as lyric_file:
        corpus = json.loads(lyric_file.read().encode('utf-8'))
    nlp = spacy.load('en_core_web_sm')
    corpus = {artist:[el for el in songlist if el['lyrics'] != 'Not Available!'] for (artist,songlist) in corpus.items()}
    deleted_artist = []
    for artist,songlist in corpus.items():   
        badsongs = []     
        for i,song in enumerate(songlist):                
            song['title'] = song['title'].replace(u"\u00A0", " ")
            song['lyrics'] = re.sub(r'\[.*?\]','',song['lyrics'])
            num_lines = sum(1 for line in song['lyrics'])                
            song['lyrics'] = re.sub(r'\s+',' ',song['lyrics'])
            song['lyrics'] = song['lyrics'].lower()
            songlyrics = nlp(song['lyrics'])
            pos_counts = dict(Counter([token.pos_ for token in songlyrics]))
            song['pos_counts'] = pos_counts
            if len(song['lyrics'].split()) > 2000:
                badsongs.append(i)
            song['avg_linelen'] = len(song['lyrics'].split()) / num_lines
        for el in badsongs:
            del songlist[el]
        if len(songlist) < 20:
            deleted_artist.append(artist)
        print(artist)

    for el in deleted_artist:
        del corpus[el]
    #import pdb; pdb.set_trace()
    with open('corpus_data/preprocessedf_corpus.json','w') as ofile:
        json.dump(corpus,ofile,indent=4)
    ofile.close()




if __name__ == "__main__":
    main()