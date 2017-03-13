
import networkx as nx
import gensim.models
import itertools
import numpy as np
from multiprocessing import Pool

def build_semanticnetwork(\
    text=None, \
    model=None, \
    distfunc=None, \
    distattr = 'dist', \
    num_dim=100, \
    nodeattrs=None, \
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

    # build graph
    if verbose: print('Building graph of {} nodes...'.format(len(model.vocab.keys())))
    
    G = nx.Graph() # fresh new graph

    # add nodes, apply node attr functions (ie centrality, etc)
    G.add_nodes_from(list(model.vocab.keys()))
    if nodeattrs is not None:
        for attr, attrfunc in nodeattrs.items():
            attrvals = attrfunc(G)
            nx.set_node_attributes(G,attrname,attrvals)

    # add edges, apply edge attr functions (possibly in parallel)
    '''
    edges = itertools.product(model.vocab.keys(),model.vocab.keys()) # generator
    vecpairs = [(model[u],model[v]) for (u,v) in edges] # also lazy
    if workers is not None:
        if verbose: print('starting pool..')
        with Pool(workers) as p:
            distances = p.map(distfunc,vecpairs)
        if verbose: print('pool finished..')
    else:
        distances = map(distfunc,vecpairs)
    '''
    # set edges using iterator
    itertools.product(model.vocab.keys(),model.vocab.keys())
    G.add_edges_from(edges)

    print('Calculating distances..')

    dist = dict()
    if workers is not None: p = Pool(workers)
    for ut in iter(model.vocab):
        uv = model[ut]

        # access vocab w/o copying memory
        vocab_iterator = ((model[w],uv) for w in model.vocab)
        
        # map distances to vocab combinations
        if workers is not None: p.map(distfunc, vocab_iterator, ut)
        else: map(distfunc, vocab_iterator, ut)

    distances = {(u,v):model.wv.similarity(u,v) for (u,v) in edges}
    print('Finished calculating distances..')

    edgesa, edgesb = itertools.tee(itertools.product(model.vocab.keys(),model.vocab.keys()))
    pairs = {e:v for e,v in zip(edgesb,distances)}
    nx.set_edge_attributes(G, 'dist', pairs)
    if verbose: print('finished setting edge attributes')

    # perform calculations on edges (ie flow, etc)
    if edgeattrs is not None:
        for attr, attrfunc in edgeattr.items():
            attrvals = attrfunc(G)
            nx.set_edge_attributes(G,attrname,attrvals)

    return G


def get_dist(ut,vt):
    u, uv = ut[0], ut[1].base
    v, vv = vt[0], vt[1].base

    uv = uv/np.linalg.norm(uv)
    vv = vv/np.linalg.norm(vv)

    return ((u,v), float(np.linalg.norm(uv-vv)))

def get_relations(u_vec,v_vec):
    #u_vec, v_vec = e[0],e[1]
    u_vec = u_vec/np.linalg.norm(u_vec)
    v_vec = v_vec/np.linalg.norm(v_vec)
    rel = float(np.linalg.norm(u_vec-v_vec))

    return rel

class cwtv(gensim.models.Word2Vec):
    def __init__(self, *args):

        super(gensim.models.Word2Vec, self).__init__(*args)

        self.current = low
        self.high = high

    def __next__(self):
        pass

    def __iter__(self):
        return self



    def next(self): # Python 3: def __next__(self)
        if self.current > self.high:
            raise StopIteration
        else:
            self.current += 1
            return self.current - 1

if __name__ == '__main__':

    #model = gensim.models.Word2Vec([['a','b','c'],['c','a','lol','haha'],['this','sucks','a','c']],size=3,min_count=1)
    model = gensim.models.Word2Vec.load('results/cnn.wtvmodel')
    #print(dir(model.wv.syn0))

    import time
    a = 0
    start = time.time()
    for v in iter(model.wv.syn0):
        a += np.linalg.norm(v)
        pass
    end = time.time()
    stime = end-start

    start = time.time()
    for w in model.vocab:
        #print(model[v])
        a += np.linalg.norm(model[w])
        pass
    end = time.time()
    vtime = end-start

    print(len(model.vocab))
    print(a)
    print('syn0 it took {} seconds.'.format(stime))
    print('vocab it took {} seconds.'.format(vtime))


    #for v in model.vocab.items():
    #    print(itmodel.wv.syn0[v[1].index])
    #    break


    import time
    start = time.time()
    G = build_semanticnetwork(model=model, distfunc=get_dist, workers=20, verbose=True)
    end = time.time()
    print('Took {} seconds.'.format(end-start))