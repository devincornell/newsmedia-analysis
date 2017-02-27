
from os import walk
import networkx as nx
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import functools
from multiprocessing import Pool

def get_files(folder, extension, rejection):
    # get model filenames
    modelfiles = list()
    for (dirpath, dirnames, filenames) in walk(folder):
        for file in filenames:
            if file[-len(extension):] == extension and file[-len(rejection):] != rejection:
                modelfiles.append(folder + file)
    return modelfiles

import operator as op
def ncr(n, r):
    # stole from http://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
    r = min(r, n-r)
    if r == 0: return 1
    numer = functools.reduce(op.mul, range(n, n-r, -1))
    denom = functools.reduce(op.mul, range(1, r+1))
    return numer//denom

def pvalf(edat):
    return pval(*edat)

def pval(e,weight,ku,kv,T):
    ''' given by equations 1-3 in "Unwinding the Hairball Graph"
    weight: weight of edge in question
    ku: sum(weights of edges on u)
    kv: sum(weights of edges on v)
    T: sum of all edge weights in the graph
    '''
    max_num = 1e20
    w = int(weight);
    max_m = w + 10
    p = ku*kv/(2*T**2);
    T = int(T);

    #print(ncr(T,w))
    pval = 0
    for m in range(w,max_m):
        choose = ncr(T,m)
        if choose > max_num: choose = max_num
        pval += choose * (p**m) * (1-p)**(T-m)

    return (e,pval)

def deg(n,G,attr='weight'):
    # calculate degree of node n for weighted graph
    hood = G.neighbors(n)
    return sum([G.edge[n][v][attr] for v in hood])

def sparsify_graphfile(fname):

    src = fname.split('.')[0]

    print('Loading {}.'.format(fname))
    G = nx.read_gexf(fname)

    print('Finished loading {}.'.format(fname))
    print()

    print('Compiling pool data..')
    T = sum(nx.get_edge_attributes(G,'weight').values())
    pvals = dict()
    degrees = {u:deg(u,G) for u in G.nodes()}
    edata = [((u,v),G.edge[u][v]['weight'],degrees[u],degrees[v],T) for u in G.edge for v in G.edge[u]]
    
    print('Starting sparsification..')
    p = Pool(30) # will never use that many
    pvals = p.map(pvalf,edata)
    print('Sparsification finished.')

    nx.set_edge_attributes(G,'pval',{x[0]:x[1] for x in pvals})


    # remove (based on p-value) n edges where n = numedges*(1-edge_cutoff)
    print('Removing crappy edges..')
    edges = G.edges(data=True)
    sedges = sorted(edges,key=lambda x:x[2]['pval'])
    remove_edges = [(x[0],x[1]) for x in sedges[int(len(edges)*keep_fraction):]]
    G.remove_edges_from(remove_edges)
    num_edges = len(G.edges())
    print('{}: {}% of edges retained: {} remain.'.format(src,int(num_edges/len(edges)*100),num_edges))        
    
    ofname = src + '_sparse.gexf'
    print('Writing file {}\n'.format(ofname))
    nx.write_gexf(G,ofname)
    return ofname


if __name__ == '__main__':
    keep_fraction = 0.01 # fraction of edges to keep

    if len(sys.argv) > 1:   
        results_folder = sys.argv[1]
    else:
        results_folder = 'results/'
    
    graphfiles = get_files(results_folder, '.gexf', '_sparse.gexf')

    print(graphfiles)
    print()
    
    #p = Pool(len(graphfiles))
    print(list(map(sparsify_graphfile, graphfiles)))

    #for file in graphfiles:
        #graphs[file.split('.')[0]] = nx.read_gexf(file)

    exit()



