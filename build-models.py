
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
    NDIM = 50
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
                sents.append( [w.text.lower() for w in sent if w.is_alpha] )
        print(len(sents), 'sentences.')

        print('training wtv model')
        model = gensim.models.Word2Vec(sents, size=NDIM, workers=CORES, min_count=2, sg=1)
        print('done training model')
        models[srcname] = model
        print(models)
        
    print('saving file')
    with open('results/word_models.pic', 'wb') as f:
        pickle.dump(models, f)

        