
import networkx as nx
import gensim.models
import itertools
import numpy as np
from multiprocessing import Pool
from functools import partial
import time

def build_semanticnetwork(\
    text=None, \
    model=None, \
    distfunc=None, \
    distattr = 'dist', \
    num_dim=100, \
    nodeattrs=None, \
    usenodes = None, \
    edgeattrs=None, \
    workers=None,
    verbose=False \
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

    # build graph
    if verbose: print('Building graph of {} nodes...'.format(len(usenodes)))
    
    if verbose: start = time.time()

    G = nx.Graph() # fresh new graph

    # add nodes, apply node attr functions (ie centrality, etc)
    G.add_nodes_from(usenodes)
    if nodeattrs is not None:
        for attr, attrfunc in nodeattrs.items():
            attrvals = attrfunc(G)
            nx.set_node_attributes(G,attrname,attrvals)

    # set edges using iterator
    edges = itertools.product(usenodes,usenodes)
    G.add_edges_from(edges)

    if verbose: print('Calculating distances..')
    if workers is not None: p = Pool(workers)
    for u in iter(usenodes):
        uv = model[u]

        # access vocab w/o copying memory
        vv_it = (model[v] for v in model.vocab)

        # map distances to vocab combinations
        if workers is not None: edgeweights = p.starmap(distfunc, zip(vv_it,itertools.repeat(uv)))
        else: edgeweights = map(distfunc, vv_it, uv)

        edges = ((u,v) for v in usenodes)
        nx.set_edge_attributes(G,distattr,{e:w for e,w in zip(edges,edgeweights)})
    
    if verbose: print('Finished calculating distances..')
    if verbose: end = time.time()
    if verbose: print('Took {} seconds.'.format(end-start))

    return G


def get_dist(u_vec,v_vec):
    #u_vec, v_vec = e[0],e[1]
    u_vec = u_vec/np.linalg.norm(u_vec)
    v_vec = v_vec/np.linalg.norm(v_vec)
    dist = float(np.linalg.norm(u_vec-v_vec))

    return dist


if __name__ == '__main__':

    model = gensim.models.Word2Vec([['a','b','c'],['c','a','lol','haha'],['this','sucks','a','c']],size=3,min_count=1)
    model = gensim.models.Word2Vec.load('results/cnn.wtvmodel')

    usenodes = model.vocab
    
    settings = { \
        'model':model, \
        'distfunc':get_dist, \
        'usenodes':usenodes, \
        'verbose':True \
        }
    G = build_semanticnetwork(**settings)
