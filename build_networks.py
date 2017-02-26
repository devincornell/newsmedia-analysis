

print('Make sure you run build_models.py before running this thing. They\'l use the models in the results/ folder.')
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

    # decide which words to use based on frequency of appearance in all documents
    topnwords = set()
    for src,dat in models.items():
        candidates = set([x[0] for x in dat['wordfreq'].most_common(n)])
        # remove candidates that aren't in all vocabs?
        topnwords |= candidates

    # look through each model to check vocab size
    for src, dat in models.items():
        srcvocab = set(dat['model'].vocab.keys())
        print('Building {} graph...'.format(src))
        
        G = nx.Graph()
        for v in topnwords:
            if v in srcvocab:
                G.add_node(v,freq=dat['wordfreq'][v])

        for u in G.nodes():
            for v in G.nodes():
                rel_dict = get_relations(dat['model'][u],dat['model'][v])
                G.add_edge(u,v,rel_dict)

        print('Saving {}{}.gexf file'.format(results_folder,src))
        nx.write_gexf(G,results_folder + src + '.gexf')

        print()




