# Rap Artist Lyrics Classifier

> Done as part of the final project for the University of Michigan's EECS 486 class.

The aim of this project to explore the linguistically unique nature of rap lyrics and the effectiveness of various classification models in matching those lyrics to the respective artists. The project uses a popularity-based seedlist of 44 rappers, along with information and lyrics of their respective top 50 songs, requested from Genius through its API. We then train the various classification models, using a 80-20 train-test split, on different combinations of the data and evaluated according to the recall metrics to determine the effectiveness of each combination. 

The web app then allows users to input their own lyrics or strings of text and the preferred classification model, then outputs the rap artist with the closest matching lyrics features.

A detailed summary of the project, methodologies, evaluation, and results can be found in the [PDF report](https://github.com/tansawit/rap-artist-match/blob/master/rapper-lyrics-matching-report.pdf).

**Contributors:** Thomas Horak, Brian Guo, Krishan Amin, Sawit Trisirisatayawong, Vraj Desai
