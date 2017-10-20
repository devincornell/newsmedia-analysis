

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



## basic utilities

def getfilenames(folder, model_extension):
    # get model filenames
    files = dict()
    modelfiles, wordfreqfiles = list(), list()
    for (dirpath, dirnames, filenames) in walk(folder):
        for file in filenames:
            if file[-len(model_extension):] == model_extension:
                srcname = file.split('.')[0]
                files[srcname] = folder + file
    return files


if __name__ == "__main__":

    ## SETTINGS
    # file settings
    models_folder = 'results/wtvmodels/'
    model_extension = '.wtvmodel'

    files = getfilenames(models_folder, model_extension)
    print('found {} files.'.format(len(files)))
    
    #files = {'cbsnews_pars':files['cbsnews_pars'],}
    for src, fname in files.items():
        print(src)
        model = gensim.models.Word2Vec.load(fname)
        #usenodes = [w for w in model.wv.vocab.keys() if model.wv.vocab[w].count > 40]
        usenodes = list(model.wv.vocab.keys())
        print('using {} words for matrix.'.format(len(usenodes)))
        S = sa.build_semanticmatrix(model, usenodes, verbose=True)
        print(S.shape)
        print(pd.isnull(S).sum().sum())
        
        for p in [i/10 for i in range(10)]+[0.99,]:
            topic = sa.centralized_randomwalk('trump', matrix=S, returnprob=p, max_iter=1000)
            print(p)
            print([t[0] for t in topic[:10]])
            print()
            
        print(model.most_similar('trump',topn=10))

