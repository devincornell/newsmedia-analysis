
import networkx as nx
import semanticnetwork as sn
import math
import pickle
import community

if __name__ == '__main__':

    print(dir(community.best_partition))
    # this script uses the community library
    # http://perso.crans.org/aynaud/communities/api.html
    folder = 'results/'
    extension = '_full.gexf'
    srcnames = ['nytimes', 'breitbart', 'cbsnews', 'cnn', 'foxnews', 'huffpo', 'reuters', 'usatoday', 'wapo', 'watimes']
    res = 0.8

    partitions = dict()
    for src in srcnames:
        print('Reading', src, 'graph.')
        G = nx.read_gexf(folder + src + extension)
        
        print('Calculating clusters..')
        partitions[src] = community.best_partition(G, resolution=res)
        print()

    print('Saving files..')
    with open(folder + 'cluster_partitions.pickle', 'wb') as f:
        pickle.dump(partitions,f)

