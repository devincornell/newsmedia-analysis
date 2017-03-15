

print('Make sure you run build_models.py before running this thing. They\'ll use the models in the results/ folder.')
print()

import networkx as nx
import gensim.models
from os import walk
import numpy as np
import pickle
import sys

import semanticnetwork as sn




## basic utilities

def getfilenames(results_folder, model_extension, wf_extension):
    # get model filenames
    files = dict()
    modelfiles, wordfreqfiles = list(), list()
    for (dirpath, dirnames, filenames) in walk(results_folder):
        for file in filenames:
            if file[-len(model_extension):] == model_extension:
                srcname = file.split('.')[0]
                files[srcname] = dict()
                files[srcname]['model'] = results_folder + file
                files[srcname]['wordfreq'] = results_folder + srcname + wf_extension
    return files



if __name__ == "__main__":
    ## SETTINGS
    # file settings
    results_folder = 'results/'
    model_extension = '.wtvmodel'
    wf_extension = '_wordfreq.pickle'

    # frequency settings
    freq_cutoff = 5 # min number of appearances in each source for a word

    # reduction/sparsification settings
    drop_nodes = True
    num_nodes_retained = 30 # number of most central nodes to keep
    sparsify_edges = False
    sparsify_retain_ratio = 0.3 # percentage of edges to keep
    drop_edges = True
    fraction_edges_retained = 0.3 # percentage of edges to keep (after sparsifying, if nessecary)
 

    ## CODE STARTS
    # get filenames
    if len(sys.argv) > 1:
        results_folder = sys.argv[1]
        print('Using results folder {}.'.format(results_folder))
        print()
    files = getfilenames(results_folder, model_extension, wf_extension)


    # load wordfreq files to decide which nodes to use
    wordfreqs = list()
    for src in files.keys():
        print(files[src]['wordfreq'])
        with open(files[src]['wordfreq'], 'rb') as f:
            wf = pickle.load(f)
        print('found', len(wf.keys()), 'words.')
        wordfreqs.append([w for w in wf.keys() if wf[w] > freq_cutoff])

    
    # find common set of words in each reduced vocabulary
    print('Finding common set of words.')
    nodeset = sn.common_set(wordfreqs)
    print('Keeping {} nodes appear at least {} times in all sources.'.format(len(nodeset), freq_cutoff))
    print()

    # look through each model to check vocab size

    for src in files.keys():
        modelf = files[src]['model'] # shallow copy
        model = gensim.models.Word2Vec.load(modelf)

        print('Loaded model for {}.'.format(src))

        print('Building {} graph...'.format(src))
        settings = {
            'model': model,
            'usenodes': nodeset, 
            'verbose': True, 
            'nodeattrs': { 
                'eigcent': lambda x: nx.eigenvector_centrality(x,1000,tol=1e-4),
                },
            }
        G = sn.build_semanticnetwork(**settings)
        print(len(G.nodes()), 'in', src)

        if drop_nodes:
            print('Now removing nodes that are least central: keeping {} nodes.'.format(num_nodes_retained))
            G = sn.drop_nodes(G,'eigcent', number=num_nodes_retained, keep_largest=True, verbose=True)


        if sparsify_edges:
            print('Now sparsifying edges..')

            settings = {
                'G': G,
                'keep_fraction': sparsify_retain_ratio, # percentage of edges to keep
                'weight_attr': 'weight', # used for cutoff
                'pval_attr': 'p-val', # added attr for stat. sig.
                'processes': 16, # split up the work
                'verbose':True,
                }
            G = sn.sparsify_edges_prefattach(**settings)

        if drop_edges:
            print('Keeping {}% of edges w/ highest weight..'.format(100*fraction_edges_retained))
            G = sn.drop_edges(G,'weight', fraction=fraction_edges_retained, keep_largest=True, verbose=True)

        print('Writing file..')
        nx.write_gexf(G, results_folder + src + '_sparse.gexf')
        print()
