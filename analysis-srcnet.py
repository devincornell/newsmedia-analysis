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
folder = 'results/srcnets/'
words = ['clinton', 'trump', 'military', 'obama']
rps = [0.1,0.3,0.5,0.7,0.9,0.99]

for word in words:
    for rp in rps:

        topics = dict()
        for src, x in matf.items():
            #model = gensim.models.Word2Vec.load(modelf[src])

            with open(matf[src], 'rb') as f:
                S = pickle.load(f)
            S.shape

            topics[src] = sa.centralized_randomwalk(word, matrix=S, returnprob=rp, max_iter=1000, verbose=False)

        G = nx.Graph()
        G.add_nodes_from(list(topics.keys()))
        for srcA, topicA in topics.items():
            for srcB, topicB in topics.items():
                betas = [5,8,10,15,20,30,40, 50, 60, 70, 100, 150, 200, 250, 400, 600]
                bvals = {str(b):float(sa.topic_relatedness(topicA,topicB,alpha=10,beta=b)) for b in betas}
                bvals['weight'] = bvals['10']
                G.add_edge(srcA,srcB,bvals)
        
        fname = folder+'{}-{}.gexf'.format(word,int(100*rp))
        print(fname)
        nx.write_gexf(G,fname)
