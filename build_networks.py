

print('Make sure you run build_models.py before running this thing. They\'ll use the models in the results/ folder.')
print()

import networkx as nx
import gensim.models
from os import walk
import numpy as np
import pickle

# calculates relation dictionary (as edge attributes) between every word pair
def get_relations(u_vec, v_vec):
    rel = dict()

    rel['l2_dist'] = float(np.linalg.norm(u_vec-v_vec))

    return rel


if __name__ == "__main__":
    results_folder = 'results/'
    model_extension = '.wtvmodel'
    wf_extension = '_wordfreq.pickle'
    n = 100 # top n words to keep from each source
    edge_cutoff = 1/10 # fraction of edges to keep in saved network
    central_nodes = 30 # number of most central nodes to keep
    
    print()

    # get model filenames
    modelfiles = list()
    for (dirpath, dirnames, filenames) in walk(results_folder):
        for file in filenames:
            if file[-len(model_extension):] == model_extension:
                modelfiles.append(file)

    # load models into models where models['srcname']['model'] = wtvmodel
    models = dict()
    for file in modelfiles:
        srcname = file.split('.')[0]
        models[srcname] = dict()
        
        # this is a genism.models.Word2Freq object
        models[srcname]['model'] = gensim.models.Word2Vec.load(results_folder + file)

        # this is a nltk.FreqDist object
        with open(results_folder + srcname + wf_extension, 'rb') as f:
            models[srcname]['wordfreq'] = pickle.load(f)

    # decide which words to use based on frequency of appearance
    candidset = set()
    srcvocabs = list()
    for src,dat in models.items():
        candidset |= set([x[0] for x in dat['wordfreq'].most_common(n)])
        srcvocabs.append(set(dat['model'].vocab.keys()))

    # remove words that don't appear in all sources
    removeset = set()
    for w in candidset:
        removeword = False
        for srcvocab in srcvocabs:
            if w not in srcvocab:
                removeword = True
        if removeword:
            removeset.add(w)

    # keep only words that appear in all vocabularies
    nodeset = candidset - removeset
    print('Keeping {} nodes that appear in all sources.'.format(len(nodeset)))
    print()

    # look through each model to check vocab size
    graphs = dict()
    for src, dat in models.items():
        print(src)
        srcvocab = set(dat['model'].vocab.keys())
        print('Building {} graph...'.format(src))
        
        G = nx.Graph()
        for v in nodeset:
            if v in srcvocab:
                G.add_node(v,freq=dat['wordfreq'][v])

        for u in G.nodes():
            for v in G.nodes():
                if u != v: # no self-loops
                    rel_dict = get_relations(dat['model'][u],dat['model'][v])
                    G.add_edge(u,v,rel_dict)


        # add attributes to complete graph for analysis later
        weights = {(u,v):1/(G.edge[u][v]['l2_dist']**2) for u in G.edge for v in G.edge[u]}
        nx.set_edge_attributes(G,'weight', weights)

        eig_cent = nx.eigenvector_centrality(G,max_iter=int(1e4),tol=1e-4,weight='weight')
        nx.set_node_attributes(G,'eig_cent', eig_cent)

        graphs[src] = G
    print()

    # get set of nodes to keep
    keep_nodes = set()
    for src in graphs.keys():
        sort_nodes = sorted(graphs[src].nodes(data=True),key=lambda x:x[1]['eig_cent'])
        keep_nodes |= set([n[0] for n in sort_nodes[-central_nodes:]])
    
    print('{} nodes found from all sources for comparison.'.format(len(keep_nodes)))
    print()

    # remove all but most central nodes and a percentage of edges from those
    for src in graphs.keys():
        rm_nodes = set(graphs[src].nodes()) - keep_nodes
        graphs[src].remove_nodes_from(rm_nodes)

        # remove weakest n edges where n = numedges*(1-edge_cutoff)
        print(len(graphs[src].edges()))
        edges = graphs[src].edges(data=True)
        sedges = sorted(edges,key=lambda x:x[2]['l2_dist'])
        remove_edges = [(x[0],x[1]) for x in sedges[int(len(edges)*edge_cutoff):]]
        graphs[src].remove_edges_from(remove_edges)
        num_edges = len(graphs[src].edges())
        print('{}: {}% of edges retained: {} remain.'.format(src,int(num_edges/len(edges)*100),num_edges))

        # calculate new statistics on partial graph

        # visualization parameters
        # cytoscape uses viz_size, viz_transparency, viz_color
        #viz_size = {n:v*200 for n,v in eig_cent.items()}
        #nx.set_node_attributes(graphs[src],'viz_size', viz_size)


        # save .gexf file
        print('Saving {}{}.gexf file'.format(results_folder,src))
        nx.write_gexf(graphs[src],results_folder + src + '.gexf')

        print()




