
from os import walk
import itertools
import functools
import pickle
import networkx as nx
import semanticnetwork as sn

srcnames = ['nytimes', 'breitbart', 'cbsnews', 'cnn', 'foxnews', 'huffpo', 'reuters', 'usatoday', 'wapo', 'watimes']

if __name__ == "__main__":

    folder = 'results/'
    extension = '_full.gexf'
    num_nodes_retained = 20

    fnames = [folder + sn + extension for sn in srcnames]

    ## Look at sparse graphs to get top nodes
    nodesets = list()
    for src,fname in zip(srcnames,fnames):
        print('Loading network', src)
        G = nx.read_gexf(fname)

        print('Now removing nodes that are least central: keeping {}', num_nodes_retained)
        G = sn.drop_nodes(G,'eigcent', number=num_nodes_retained, keep_largest=True, verbose=True)
        nodesets.append(set(G.nodes()))

    # reduce using set union
    keepnodes = functools.reduce(lambda x,y: x | y, nodesets)
    print('Keeping {} nodes that are top {} in at least one source.'.format(len(keepnodes), num_nodes_retained))

    print()

    print('Removing unneeded nodes.')
    smallgraphs = dict()
    for src,fname in zip(srcnames,fnames):
        print('Loading network', src)
        G = nx.read_gexf(fname)

        smallgraphs[src] = G.remove_nodes_from(set(G.nodes()) - keepnodes)

    print('Saving small graph data.')
    with open(folder + 'smallgraphs.pickle', 'wb') as f:
        pickle.dump(smallgraphs, f)





