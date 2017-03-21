
import networkx as nx
import semanticnetwork as sn
import math
import pickle

mftnames = {
    1: 'HarmVirtue',
    2: 'HarmVice',
    3: 'FairnessVirtue',
    4: 'FairnessVice',
    5: 'IngroupVirtue',
    6: 'IngroupVice',
    7: 'AuthorityVirtue',
    8: 'AuthorityVice',
    9: 'PurityVirtue',
    10: 'PurityVice',
    11: 'MoralityGeneral',
}


def getmftnodes(nodeattr):
    nodedict = {x:list() for x in mftnames.keys()} # moral foundation -> list(nodes)
    for m in mftnames.keys():
        for n,v in nodeattr.items():
            if str(m) in v.split(','):
                nodedict[m].append(n)
    return nodedict


if __name__ == '__main__':
    folder = 'results/'
    extension = '.gexf'
    srcnames = ['breitbart', 'cbsnews', 'cnn', 'foxnews', 'nytimes']

    mftscores = {s:{m:0 for m in mftnames.keys()} for s in srcnames}
    for src in srcnames:
        print('Reading', src, 'graph.')
        G = nx.read_gexf(folder + src + extension)
        print('Calculating clusters..')
        
        #first compute the best partition
        partitions = community.best_partition(G)

    with open(folder + 'cluster_partitions.pickle', 'wb') as f:
        pickle.dump(partitions,f)

