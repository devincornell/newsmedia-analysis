

print('Make sure you run build_models.py before running this thing. They\'ll use the models in the results/ folder.')
print()

import networkx as nx
import gensim.models
from os import walk
import numpy as np
import pickle
import sys

import semanticnetwork as sn



if __name__ == "__main__":
    # file settings
    results_folder = 'results/'
    model_extension = '.wtvmodel'
    wf_extension = '_wordfreq.pickle'

    # frequency settings
    freq_cutoff = 5

    # reduction/sparsification
    remove_all_but_central = False
    num_nodes_retained = 30 # number of most central nodes to keep
    remove_weakest_edges = True
    edge_retain_ratio = 0.1
    

    if len(sys.argv) > 1:
        results_folder = sys.argv[1]
        print('Using results folder {}.'.format(results_folder))
        print()

    files = getfilenames(results_folder, model_extension, wf_extension)


    # load wordfreq files to decide which nodes to use
    #wordfreqs = dict()
    wordfreqs = list()
    for src in files.keys():
        print(files[src]['wordfreq'])
        with open(files[src]['wordfreq'], 'rb') as f:
            wf = pickle.load(f)
        print('found', len(wf.keys()), 'words.')
        wordfreqs.append([w for w in wf.keys() if wf[w] > freq_cutoff])

    # skipping 
    
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
        print('Writing file..')
        G.write_gexf(results_folder + src + '.gexf')



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
