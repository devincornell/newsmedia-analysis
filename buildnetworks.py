

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
    matrix_folder = 'results/mat_medium10/'
    
    model_extension = '.wtvmodel'

    if len(sys.argv) > 1:
        workers = int(sys.argv[1])
    else:
        workers = 10

    files = getfilenames(models_folder, model_extension)
    print('found {} files.'.format(len(files)))

    #files = {'cbsnews_pars':files['cbsnews_pars']}
    for src, fname in files.items():
        print(src)
        model = gensim.models.Word2Vec.load(fname)
        usenodes = list(model.wv.vocab.keys())
        usenodes = [w for w in model.wv.vocab.keys() if model.wv.vocab[w].count > 10]
        print('using {} words for matrix.'.format(len(usenodes)))
        S = sa.build_semanticmatrix(model, usenodes, verbose=True, workers=workers)
        print('built matrix for', src)
        print('now saving')
        
        #with open(matrix_folder + src + '.mat', 'wb') as f:
        #    pickle.dump(S, f)
        S.to_hdf(matrix_folder+src+'.hdf', key='S')
            
        print(src, 'now done.')
    
    print('ddfd')
    
    
    


