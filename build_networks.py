

print('Make sure you run build_models.py before running this thing. They\'l use the models in the results/ folder.')


import gensim.models

from os import walk

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

    # look through each model
    for src, model in models.items():
        print(list(model.vocab.keys())[0:10])
        #for v in model.vocab:
        #    print(v)
        #print(model.vocab)




