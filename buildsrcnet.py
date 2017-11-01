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
from scipy.stats.stats import pearsonr   



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


scale = 10
matf = getfilenames('results/mat_common/','.hdf')
folder = 'results/srcnets/'
words = ['trump', 'military']
rps = [0, 0.8, 0.999]
#betas = [10, 20, 40, 50, 200, 400, 600]


for word in words:
    for rp in rps:

        topics = dict()
        for src, x in matf.items():
            S = pd.read_hdf(matf[src], key='S')

            topics[src] = sa.centralized_randomwalk(word, matrix=S, returnprob=rp, max_iter=10000, verbose=False)

        G = nx.Graph()
        G.add_nodes_from(list(topics.keys()))
        for srcA, topicA in topics.items():
            for srcB, topicB in topics.items():
                r2, pval = sa.topic_correlation(topicA[1:],topicB[1:])
                eweights = dict()
                eweights['weight'] = float(r2)
                eweights['pval'] = float(pval)
                G.add_edge(srcA,srcB,eweights)
        
        f = '{}-{}'.format(word,int(1000*rp))
        fname = folder + f + '.gexf'
        print(fname)
        nx.write_gexf(G,fname)
        df = pd.DataFrame(nx.to_numpy_matrix(G), index=G.nodes(), columns=G.nodes())
        df.to_csv('results/srcnets/'+f+'.csv')

