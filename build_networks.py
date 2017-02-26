

print('Make sure you run build_models.py before running this thing. They\'l use the models in the results/ folder.')


import networkx as nx
import gensim.models
from os import walk
import numpy as np

# calculates relation dictionary (as edge attributes) between every word pair
def get_relations(u_vec, v_vec):
    rel = dict()

    rel['l2_dist'] = np.linalg.norm(u_vec-v_vec)

    return rel


if __name__ == "__main__":
    model_folder_name = 'results/'
    model_extension = '.wtvmodel'
    
    # get model filenames
    modelfiles = list()
    for (dirpath, dirnames, filenames) in walk(model_folder_name):
        for file in filenames:
            if file[-len(model_extension):] == model_extension:
                modelfiles.append(file)


    # load models into models where models['srcname'] = wtvmodel
    models = dict()
    for file in modelfiles:
        models[file.split('.')[0]] = gensim.models.Word2Vec.load(model_folder_name + file)

    # look through each model to check vocab size
    for src, model in models.items():
        vocab = list(model.vocab.keys())
        print('{} contains {} words.'.format(src,len(vocab)))
        print('Building {} graph...'.format(src))
        
        G = nx.Graph()
        for v in vocab:
            G.add_node(v)
        for u in G.nodes():
            for v in G.nodes():
                rel_dict = get_relations(model[u],model[v])
                G.add_edge(u,v,rel_dict)

        print('Saving {}{}.gexf file'.format(model_folder_name,src))
        nx.write_gexf(G,model_folder_name + src + '.gexf')




