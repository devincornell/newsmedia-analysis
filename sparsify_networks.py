
from os import walk
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

def get_files(folder, extension):
    # get model filenames
    modelfiles = list()
    for (dirpath, dirnames, filenames) in walk(folder):
        for file in filenames:
            if file[-len(extension):] == extension:
                modelfiles.append(folder + file)
    return modelfiles

import operator as op
def ncr(n, r):
    # stole from http://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom


def pval(weight,ku,kv,T):
    ''' given by equations 1-3 in "Unwinding the Hairball Graph"
    weight: weight of edge in question
    ku: sum(weights of edges on u)
    kv: sum(weights of edges on v)
    T: sum of all edge weights in the graph
    '''
    max_m = 500
    w = int(weight)
    pval = sum([ncr(T,m) * (p**m) * (1-p)**(T-m) for m in xrange(w:max_m)])
    return pval

def deg(n,G,attr='weight'):
    # calculate degree of node n for weighted graph
    hood = G.neighbors(n)
    return sum([G.edge[u][v][attr] for v in hood])


if __name__ == '__main__':
    keep_fraction = 0.01 # fraction of edges to keep

    if len(sys.argv) > 1:
        graphfiles = get_files(sys.argv[1], '.gexf')
    else:
        graphfiles = get_files('results/', '.gexf')

    graphs = dict()
    for file in graphfiles:
        #graphs[file.split('.')[0]] = nx.read_gexf(file)
        G = nx.read_gexf(file)

        print('Loaded {}.'.format(file))
        print()

        eweights = [G.edge[u][v]['weight'] for u in G for v in G.edge[u]]
        plt.histogram(eweights)
        plt.show()

        T = sum(nx.get_edge_attributes(G,'weight'))
        pvals = {(u,v):pval(G.edge[u][v]['weight'],deg(u,G),deg(v,G),T) for u in G.edge for v in G.edge[v]}
        nx.set_edge_attributes(G,'pval',pvals)
    exit()

    # remove weakest n edges where n = numedges*(1-edge_cutoff)
    edges = graphs[src].edges(data=True)
    sedges = sorted(edges,key=lambda x:x[2]['l2_dist'])
    remove_edges = [(x[0],x[1]) for x in sedges[int(len(edges)*edge_cutoff):]]
    graphs[src].remove_edges_from(remove_edges)
    num_edges = len(graphs[src].edges())
    print('{}: {}% of edges retained: {} remain.'.format(src,int(num_edges/len(edges)*100),num_edges))