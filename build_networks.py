

print('Make sure you run build_models.py before running this thing. They\'ll use the models in the results/ folder.')
print()

import networkx as nx
import gensim.models
from os import walk
import numpy as np
import pickle
import sys

# calculates relation dictionary (as edge attributes) between every word pair
def get_relations(u_vec, v_vec):
    rel = dict()
    u_vec = u_vec/np.linalg.norm(u_vec)
    v_vec = v_vec/np.linalg.norm(v_vec)
    rel['l2_dist'] = float(np.linalg.norm(u_vec-v_vec))

    return rel


if __name__ == "__main__":
    results_folder = 'results/'
    model_extension = '.wtvmodel'
    wf_extension = '_wordfreq.pickle'
    n = 1000 # top n words to keep from each source
    remove_all_but_central = False
    central_nodes = 30 # number of most central nodes to keep
    remove_weakest_edges = True
    edge_retain_ratio = 0.3


    if len(sys.argv) > 1:
        results_folder = sys.argv[1]
        print('Using results folder {}.'.format(results_folder))

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

    # remove some nodes from graph
    if remove_all_but_central:
        # get set of nodes to keep
        keep_nodes = set()
        for src in graphs.keys():
            sort_nodes = sorted(graphs[src].nodes(data=True),key=lambda x:x[1]['eig_cent'])
            keep_nodes |= set([n[0] for n in sort_nodes[-central_nodes:]])

        # remove all but most central nodes
        for src in graphs.keys():
            rm_nodes = set(graphs[src].nodes()) - keep_nodes
            graphs[src].remove_nodes_from(rm_nodes)

        print('{} nodes found from all sources for comparison.'.format(len(keep_nodes)))
        print()

    else:
        print('Keeping all nodes from each source.')
        # calculate new statistics on partial graph

    for src in graphs.keys():
        # save .gexf file
        print('Saving {}{}.gexf file'.format(results_folder, src))
        nx.write_gexf(graphs[src], results_folder + src + '.gexf')

        print()

    #creatin n' savin' sparsified graphs
    for src in graphs.keys():
        if edge_retain_ratio:
            # remove (based on p-value) n edges where n = numedges*(1-edge_cutoff)
            print('Removing weak edges..')
            edges = G.edges(data=True)
            sedges = sorted(edges,key=lambda x:x[2]['weight'])
            keep_num = int(len(edges)*edge_retain_ratio)
            remove_edges = [(x[0],x[1]) for x in sedges[-keep_num:]]
            G.remove_edges_from(remove_edges)
            num_edges = len(G.edges())
            print('{}: {}% of edges retained: {} remain.'.format(src,int(num_edges/len(edges)*100),num_edges))
            ofname = src + '_sparse.gexf'
            print('Writing file {}\n'.format(ofname))
            nx.write_gexf(G, ofname)

        else:
            print('Retaining all edges.')

    # visualization parameters
    # cytoscape uses viz_size, viz_transparency, viz_color
    # print('Applying visualization attributes.\n')
    # for src in graphs.keys():
    # eig_cent = nx.get_node_attributes(G,'eig_cent')
    # viz_size = {n:v*200 for n,v in eig_cent.items()}
    # nx.set_node_attributes(graphs[src],'viz_size', viz_size)
