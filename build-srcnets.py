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


scale = 10
centerwords = ['NN',]
rps = [0,]

for centword in centerwords:
    print('centerword', centword)
    for rp in rps:
        print('return prob', rp)
        
        print('running randomwalks')
        dists = dict()
        store = pd.HDFStore('results/pos_networks.h5')
        networks = {src:store.get(src) for src,net in store.items()}
        for src, S in networks.items():
            #print(S.index[:5])
            dists[src] = sa.centralized_randomwalk(centword, matrix=S, returnprob=rp, max_iter=10000, verbose=False)
            print(dists[src][40:])

        print('building networks')
        G = nx.Graph()
        G.add_nodes_from(dists.keys())
        for u, utopic in dists.items():
            for v, vtopic in dists.items():
                if u != v:
                    d = sa.topic_similarity(utopic,vtopic)
                    eweights = dict()
                    eweights['weight'] = float(d)
                    G.add_edge(u,v,eweights)
        
        # normalize weights
        weights = np.array([e['weight'] for u,v,e in G.edges(data=True)])
        #weights = (weights-min(weights))/(max(weights)-min(weights))
        for u,v in G.edges():
            G.edge[u][v]['weight'] = (G.edge[u][v]['weight']-min(weights))/(max(weights)-min(weights))
        
        folder = 'results/srcnets/'
        fname = '{}-{}'.format(centword,int(1000*rp))
        fpath = folder + fname + '.gexf'
        print(fname)
        nx.write_gexf(G,fpath)
        df = pd.DataFrame(nx.to_numpy_matrix(G), index=G.nodes(), columns=G.nodes())
        df.to_csv(fpath+'.csv')

