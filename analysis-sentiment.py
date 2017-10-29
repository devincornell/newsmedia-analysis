import networkx as nx
import gensim.models
from os import walk
import numpy as np
import pandas as pd
import pickle
import sys
import re
import semanticanalysis as sa
import functools
import itertools
from multiprocessing import Pool
import sys
import pickle
from nltk.corpus import sentiwordnet as swn



def getfilenames(folder, file_extension):
    # get model filenames
    files = dict()
    modelfiles, wordfreqfiles = list(), list()
    for (dirpath, dirnames, filenames) in walk(folder):
        for file in filenames:
            if file[-len(file_extension):] == file_extension:
                srcname = file.split('.')[0]
                files[srcname] = folder + file
    return files


scale = 100
matf = getfilenames('results/mat_small/','.mat')
modelf = getfilenames('results/wtvmodels/','.wtvmodel')

for src, x in matf.items():
    model = gensim.models.Word2Vec.load(modelf[src])

    with open(matf[src], 'rb') as f:
        S = pickle.load(f)
    S.shape


    topic = sa.centralized_randomwalk('trump', matrix=S, returnprob=0.1, max_iter=1000, verbose=False)

    score = 0
    for w, s in topic[1:]:
        freq = model.wv.vocab[w].count
        
        sentiment = sum([sent.pos_score()-sent.neg_score() for sent in swn.senti_synsets(w)])
        score += sentiment*s*scale

    print(src, score)
