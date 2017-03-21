
import networkx as nx
import semanticnetwork as sn
import math
import pickle
import community

if __name__ == '__main__':
    # this script uses the community library
    # http://perso.crans.org/aynaud/communities/api.html
    folder = 'results/'
    extension = '.gexf'
    srcnames = ['breitbart', 'cbsnews', 'cnn', 'foxnews', 'nytimes']

    partitions = dict()
    for src in srcnames:
        print('Reading', src, 'graph.')
        G = nx.read_gexf(folder + src + extension)
        
        print('Calculating clusters..')
        partitions[src] = community.best_partition(G)
        print()

    print('Saving files..')
    with open(folder + 'cluster_partitions.pickle', 'wb') as f:
        pickle.dump(partitions,f)

