
from os import walk
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_files(folder, extension):
    # get model filenames
    modelfiles = list()
    for (dirpath, dirnames, filenames) in walk(folder):
        for file in filenames:
            if file[-len(extension):] == extension:
                modelfiles.append(folder + file)
    return modelfiles


if __name__ == '__main__':

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
    exit()

    # remove weakest n edges where n = numedges*(1-edge_cutoff)
    edges = graphs[src].edges(data=True)
    sedges = sorted(edges,key=lambda x:x[2]['l2_dist'])
    remove_edges = [(x[0],x[1]) for x in sedges[int(len(edges)*edge_cutoff):]]
    graphs[src].remove_edges_from(remove_edges)
    num_edges = len(graphs[src].edges())
    print('{}: {}% of edges retained: {} remain.'.format(src,int(num_edges/len(edges)*100),num_edges))