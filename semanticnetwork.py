
import networkx as nx
import gensim.models
import itertools
import numpy as np
from multiprocessing import Pool

def build_semanticnetwork(\
    text=None, \
    model=None, \
    distfunc=None, \
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
    print('Calculating distances..')
    edges = itertools.product(model.vocab.keys(),model.vocab.keys())
    distances = {(u,v):model.wv.similarity(u,v) for (u,v) in edges}
    print('Finished calculating distances..')

    edgesa, edgesb = itertools.tee(itertools.product(model.vocab.keys(),model.vocab.keys()))
    G.add_edges_from(edgesa)
    pairs = {e:v for e,v in zip(edgesb,distances)}
    nx.set_edge_attributes(G, 'dist', pairs)
    if verbose: print('finished setting edge attributes')

    # perform calculations on edges (ie flow, etc)
    if edgeattrs is not None:
        for attr, attrfunc in edgeattr.items():
            attrvals = attrfunc(G)
            nx.set_edge_attributes(G,attrname,attrvals)

    return G

'''
def save_model(model=None, wordattr=None, verbose=False):
    if verbose: print('Saving model..')

    if model is not None:
        # save model and word frequency count
        if verbose: print('Saving {} and {}.'format(srcname, srcname))
        model.save(model_file)
        
    if wordattr is not None:
        with open(wordattr_file,'wb') as f:
            pickle.dump(freq_dist, f)
        if verbose: print()


def open_model(model_file=None, wordattr_file=None, wordattr_file=None, verbose=False):
    ' Opens a model file and a word attribute dictionary.
    
    model_file: should bea file saved from .save_model() or
    gensim.models.Word2Vec.save().

    wordattr_file: dictionary of words and attributes (like frequency)
    '


    dat = dict()
    if model is not None:
        if verbose: print('Loading model from {}'.format(model_file))
        # this is a genism.models.Word2Freq object
        models[srcname]['model'] = gensim.models.Word2Vec.load(results_folder + file)
            
    if wordattr is not None:
        # this is a nltk.FreqDist object
        with open(results_folder + srcname + wf_extension, 'rb') as f:
            dat['wordfreq'] = pickle.load(f)

'''

def get_relations(e):
    u_vec, v_vec = e[0],e[1]
    rel = dict()
    u_vec = u_vec/np.linalg.norm(u_vec)
    v_vec = v_vec/np.linalg.norm(v_vec)
    rel['l2_dist'] = float(np.linalg.norm(u_vec-v_vec))

    return rel['l2_dist']

if __name__ == '__main__':

    #model = gensim.models.Word2Vec([['a','b','c'],['c','a','lol','haha'],['this','sucks','a','c']],size=3,min_count=1)
    model = gensim.models.Word2Vec.load('results/cnn.wtvmodel')
    import time
    start = time.time()
    G = build_semanticnetwork(model=model, distfunc=get_relations, workers=20, verbose=True)
    end = time.time()
    print('Took {} seconds.'.format(end-start))