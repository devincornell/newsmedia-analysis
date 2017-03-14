
import networkx as nx
import gensim.models
import itertools
import numpy as np
from multiprocessing import Pool
import time
import functools



##### Semantic Network Construction #####

def get_cosdist(u_vec,v_vec):
    #u_vec, v_vec = e[0],e[1]
    u_vec = u_vec/np.linalg.norm(u_vec)
    v_vec = v_vec/np.linalg.norm(v_vec)
    dist = float(np.linalg.norm(u_vec-v_vec))

    return dist

def calc_edgeweights(G, beta=2):
    weights = {(u,v):G.edge[u][v]['dist']**(-beta) for (u,v) in G.edges()}
    return weights

def build_semanticnetwork(
    text = None, 
    model = None, 
    usenodes = None, 
    distfunc = get_cosdist, 
    distattr = 'dist', 
    num_dim = 100, 
    nodeattrs = None, 
    edgeattrs = {'weight': calc_edgeweights}, 
    workers = None,
    verbose = False 
    ):
    ''' This function will build a semantic network from either:
    a) a set of text blocks (sentences, paragraphs, documents, etc)
        (if text is defined)
    or b) a pre-generated gensim.models.Word2Vec object
        (if model is not defined)

    text: format=[[<sent/par/docs>],] If provided, will first
    build a word2vec model from words then apply them to 
    nodes in output network.
    
    model: a gensim.models.Word2Vec object to be converted into
    a network. NOTE, if text is defined this will be ignored.

    relationfunc: a function that takes two vectors and returns
    a distance measurement from them

    nodeattrs: a dictionary of attribute, functions where the function
    takes a graph and outputs a dictionary of node, value pairs that
    should be applied to the network

    edgeattrs: a dictionary of attribute, functions where the function
    takes a graph and outputs a dictionary of node, value pairs that
    should be applied to the network

    verbose: will "talk out loud" about what it's doing.

    '''

    # catch error cases (neither or both text,model are defined)
    if (text is None and model is None) or (text is not None and model is not None):
        raise(Exception('One of (text, model) needs to be defined!'))
    
    # build model
    if text is not None:
        if verbose: print("Training model on {} wordgroups.".format(len(text)))
        model = gensim.models.Word2Vec(text, size=num_dim, workers=20)
        if verbose: print('model contains {} words.'.format(len(model.vocab.keys())))

    # use only these nodes
    if usenodes is None:
        usenodes = model.vocab.keys()
        for n in usenodes:
            if not n in model.vocab.keys():
                raise(Exception('Not all usenodes are in the model dict.'))

    # build graph
    if verbose: print('Building graph of {} nodes...'.format(len(usenodes)))
    
    if verbose: start = time.time()

    G = nx.Graph() # fresh new graph

    # add nodes, apply node attr functions (ie centrality, etc)
    G.add_nodes_from(usenodes)    

    if verbose: print('Calculating distances..')
    for i in range(len(usenodes)):
        u = usenodes[i]
        uv = model[u]

        # map distances to vocab combinations
        edgesa, edgesb = itertools.tee(((u,v) for v in usenodes[i+1:]))
        G.add_edges_from(edgesa)
        distances = (distfunc(uv,model[v]) for v in usenodes[i+1:])
        nx.set_edge_attributes(G,distattr,{e:w for e,w in zip(edgesb,distances)})
    
    if verbose: print('Finished calculating distances..')
    if verbose: end = time.time()
    if verbose: print('Took {} seconds to calculate distances for {} edges.'.format(end-start, len(G.edges())))

    # apply supplied functions
    if verbose: print('Calculating user functions..')
    if verbose: start = time.time()
    if nodeattrs is not None:
        for attr, attrfunc in nodeattrs.items():
            attrdict = attrfunc(G)
            nx.set_node_attributes(G, attr, attrdict)
    if edgeattrs is not None:
        for attr, attrfunc in edgeattrs.items():
            attrdict = attrfunc(G)
            nx.set_edge_attributes(G, attr, attrdict)
    if verbose: end = time.time()
    if verbose: print('Took {} seconds to apply user functions.'.format(end-start))
    return G
    


##### Sparsify Functions #####

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

def sparsify_graph(
    G=None,
    keep_fraction=0.5,
    weight_attr='weight',
    pval_attr='p-val',
    processes=16,
    verbose=False,
    ):


    if verbose: print('Compiling pool data..')
    T = sum(nx.get_edge_attributes(G,weight_attr).values())
    pvals = dict()
    degrees = {u:deg(u,G) for u in G.nodes()}

    edata = (((u,v),G.edge[u][v][weight_attr],degrees[u],degrees[v],T) for (u,v) in G.edges())
    
    if verbose: print('Starting sparsification..')
    p = Pool(processes) # will never use that many
    pvals = p.map(pvalf,edata)
    if verbose: print('Sparsification finished.')

    nx.set_edge_attributes(G,pval_attr,{x[0]:x[1] for x in pvals})

    # remove (based on p-value) n edges where n = numedges*(1-edge_cutoff)
    if verbose: print('Removing crappy edges..')
    edges = G.edges(data=True)
    sedges = sorted(edges,key=lambda x:x[2][pval_attr])
    remove_edges = [(x[0],x[1]) for x in sedges[int(len(edges)*keep_fraction):]]
    G.remove_edges_from(remove_edges)
    num_edges = len(G.edges())
    if verbose: print('{}% of edges retained: {} remain.'.format(int(num_edges/len(edges)*100),num_edges))
    
    return G


if __name__ == '__main__':

    #model = gensim.models.Word2Vec([['a','b','c'],['c','a','lol','haha'],['this','sucks','a','c']],size=3,min_count=1)
    model = gensim.models.Word2Vec.load('results/cnn.wtvmodel')

    usenodes = list(model.vocab)[:1000]
    
    settings = { 
        'model':model, 
        'usenodes':usenodes, 
        'verbose':True, 
        'nodeattrs': { 
            'eigcent': lambda x: nx.eigenvector_centrality(x,1000,tol=1e-4),
            },
        'edgeattrs': {
            'weight': calc_edgeweights,
            }, 
        }
    G = build_semanticnetwork(**settings)
    
    print('Now sparsifying..')

    settings = {
        'G': G,
        'keep_fraction': 0.01,
        'weight_attr': 'weight',
        'pval_attr': 'p-val',
        'processes': 16,
        'verbose':True,
        }
    G = sparsify_graph(**settings)


    print('Saving file..')
    nx.write_gexf(G,'results/test_sparse.gexf')
