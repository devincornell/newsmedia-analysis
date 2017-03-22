
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
        print('Calculating centrality for each mft category..')
        mftattr = nx.get_node_attributes(G, 'mft')
        mftmap = getmftnodes(mftattr)
        for moral,nodes in mftmap.items():
            eigcent = nx.get_node_attributes(G,'eigcent')
            for n in nodes:
                mftscores[src][moral] += eigcent[n]
            if len(nodes) > 0:
                mftscores[src][moral] /= len(nodes) # av eig centrality

        for moral,apl in mftscores[src].items():
            print(mftnames[moral], 'had apl', apl)

        print()

    with open(folder + 'mftcentrality.pickle', 'wb') as f:
        pickle.dump(mftscores,f)

