

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


if __name__ == "__main__":
    CORES = 10
    
    
    with open('results/pos_models.pic', 'rb') as f:
        models = pickle.load(f)
        
    m0 = list(models.keys())[0]
    postags = set(models[m0].wv.vocab)
    for src, model in models.items():
        postags &= set(model.wv.vocab)
    print('using', len(postags), 'nodes.')
    usenodes = postags
    
    for src, model in models.items():
        
        print('building', src, 'matrix with', CORES, 'cores.')
        S = sa.build_semanticmatrix(model, usenodes, verbose=True, workers=CORES)
        print('built matrix for', src)
        
        print('now saving')
        S.to_hdf('results/pos_networks.hdf', key=src)
        print(src, 'complete.')
    
    print('ddfd')
    
    
    


