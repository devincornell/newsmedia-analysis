

print('Make sure you run build_models.py before running this thing. They\'ll use the models in the results/ folder.')
print()

import networkx as nx
import gensim.models
from os import walk
import numpy as np
import pickle
import sys
import re
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


def getmftdict(file='Data/mft.dic'):
    mftdict = dict()
    with open(file, 'r') as f:
        ignore = False
        for line in f:
            if line[0] == '%':
                ignore = not ignore
            elif not ignore and len(line.split()) > 0:
                w = line.split()
                mftdict[w[0]] = [int(x) for x in w[1:]]
    return mftdict

def getmftmatches(mftdict,words):
    #mftdict = getmftdict()
    #mft = {k:','.join((str(l) for l in v)) for k,v in mftdict.items()}
    comp = [(re.compile(mstr),val) for mstr,val in mftdict.items()]
    matches = dict()
    for n in words:
        for (matchre, val) in comp:
            if matchre.match(n) is not None:
                if len(val) > 1:
                    matches[n] = ','.join((str(v) for v in val))
                else:
                    matches[n] = str(val[0])
                break
    return matches


if __name__ == "__main__":

    ## SETTINGS
    # file settings
    results_folder = 'results/'
    model_extension = '.wtvmodel'
    wf_extension = '_wordfreq.pickle'

    # frequency settings
    freq_cutoff = 5 # min number of appearances in each source for a word

    # reduction/sparsification settings
    drop_nodes = False
    num_nodes_retained = 30 # number of most central nodes to keep
    sparsify_edges = False
    sparsify_retain_ratio = 0.3 # percentage of edges to keep
    drop_edges = False
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
    mftdict = getmftdict()
    mft = {k:','.join((str(l) for l in v)) for k,v in mftdict.items()}

    # start actually building graphs
    for src in files.keys():
        modelf = files[src]['model'] # shallow copy
        model = gensim.models.Word2Vec.load(modelf)
        with open(files[src]['wordfreq'], 'rb') as f:
            wf = pickle.load(f)

        print('Loaded model for {}.'.format(src))

        print('Building {} graph...'.format(src))
        settings = {
            'model': model,
            'usenodes': nodeset, 
            'verbose': True,
            'nodeattrs': {
                'eigcent': lambda xG: nx.eigenvector_centrality(xG,1000,tol=1e-4),
                'wordfreq': lambda xG: {x:wf[x] for x in xG.nodes()},
                'mft': lambda xG: getmftmatches(mftdict, xG.nodes())
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
