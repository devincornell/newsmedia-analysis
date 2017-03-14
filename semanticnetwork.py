
import networkx as nx
import gensim.models
import itertools
import numpy as np
from multiprocessing import Pool
from functools import partial
import time


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
    

if __name__ == '__main__':

    model = gensim.models.Word2Vec([['a','b','c'],['c','a','lol','haha'],['this','sucks','a','c']],size=3,min_count=1)
    model = gensim.models.Word2Vec.load('results/cnn.wtvmodel')

    usenodes = list(model.vocab)[:1000]
    
    settings = { 
        'model':model, 
        'usenodes':usenodes, 
        'verbose':True, 
        'nodeattrs': { 
            'eigcent': nx.eigenvector_centrality,
            },
        }
    G = build_semanticnetwork(**settings)
    print('Saving file..')
    nx.write_gexf(G,'results/test.gexf')
