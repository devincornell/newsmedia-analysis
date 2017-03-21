
import networkx as nx
import semanticnetwork as sn
import math

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

def vdiff(uv,vv):
    return l2norm([u-v for u in uv for v in vv])


def l2norm(listvec):
    norm = 0
    for v in listvec:
        norm += v**2

    return math.sqrt(norm)

if __name__ == '__main__':
    folder = 'results/'
    extension = '.gexf'
    srcnames = ['breitbart', 'cbsnews', 'cnn', 'foxnews', 'nytimes']
    testword = 'trump'

    mftscores = {s:{m:0 for m in mftnames.keys()} for s in srcnames}
    print('Using testword', testword)
    for src in srcnames:
        print('Reading', src, 'graph.')
        G = nx.read_gexf(folder + src + extension)
        print('Calculating av path len for each mft category..')
        mftattr = nx.get_node_attributes(G, 'mft')
        mftmap = getmftnodes(mftattr)
        for moral,nodes in mftmap.items():
            #print(mftnames[moral], 'has', len(nodes), 'nodes.')
            for n in nodes:
                pl = 0
                pl = nx.shortest_path_length(G,testword,n,weight='dist')
                try:
                    pl = nx.shortest_path_length(G,testword,n,weight='dist')
                except:
                    pass
                mftscores[src][moral] += pl
            if len(nodes) > 0:
                mftscores[src][moral] /= len(nodes) # av path len

        for moral,apl in mftscores[src].items():
            print(mftnames[moral], 'had apl', apl)

        print()

    with open(folder + 'mftpaths_' + testword + '.pickle', 'wb') as f:
        pickle.dump(f)

