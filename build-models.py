
import gensim.models
import json
from nltk import pos_tag
import nltk
import nltk.corpus
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
import pprint
import re
from articles import *
import spacy

punctlist = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '``']

def remove_specialchars(paragraph,stopwords,specialchars):
    newpar = [w for w in paragraph if w not in stopwords]
    newpar = [w for w in newpar if w not in punctlist]
    return newpar


if __name__ == "__main__":

    # script params
    NDIM = 3
    CORES = 44
    
    articles = Articles('results/articles.db')
    models = dict()
    for source in list(articles.get(sel=['distinct source',])):
        srcname = source['distinct source']
        print(srcname)
    
        texts = [a['text'] for a in articles.get(sel=['text',], where='source == "{}"'.format(srcname))]
        print(len(texts), 'texts')
        nlp = spacy.load('en')
        #, batch_size=500, n_threads=-1)
        sents = list()
        for doc in nlp.pipe(texts, disable=['ner','textcat'], n_threads=CORES):
            for sent in doc.sents:
                sents.append( [w.tag_ for w in sent] )

        print('training wtv model')
        model = gensim.models.Word2Vec(sents, size=NDIM, workers=CORES, min_count=2, sg=1)
        print('done training model')
        models[srcname] = model
        print(models)
        
    print('saving file')
    with open('results/pos_models.pic', 'wb') as f:
        pickle.dump(models, f)
    exit()

    # remove special characters
    #src_par = [[w for w in par if w not in punctlist] for par in src_par]

    # convert to lower case
    #src_par = [[w.lower() for w in par] for par in src_par]

    # remove stopwords
    #src_par = [[w for w in par if w not in stopwords and w.isalnum()] for par in src_par]

    # keep only usenodes
    with open('results/usenodes.pic', 'rb') as f:
        usenodes = pickle.load(f)
        src_par = [[w for w in par if w in usenodes] for par in src_par]

        # train model on all sentences from source
        print('{} sentences for {}.'.format(len(src_par), srcname))
        print("Training model on {}".format(srcname))
        model = gensim.models.Word2Vec(src_par, size=num_dim, workers=44, min_count=2, sg=1)
        print('{} contains {} unique words.'.format(srcname,len(model.wv.vocab)))

        # save model and word frequency count
        print('Saving {}_pars.wtvmodel'.format(srcname))
        model.save('{}{}_pars.wtvmodel'.format(results_folder,srcname))
        print()
