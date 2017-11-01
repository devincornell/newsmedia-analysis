

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
    print()

    #files = {'cbsnews_pars':files['cbsnews_pars']}
    usenodes = set()
    for src, fname in files.items():
        print(src)
        model = gensim.models.Word2Vec.load(fname)
        nodes = set(model.wv.vocab.keys())
        
        if len(usenodes) == 0:
            usenodes = nodes
        else:
            usenodes &= nodes
        
        print('using {} words for matrix.'.format(len(nodes)))
        print()
    
    print('using {} common words for networks.'.format(len(usenodes)))
    with open('results/usenodes.pic', 'wb') as f:
        pickle.dump(usenodes, f)
    
    


