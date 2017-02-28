#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 09:34:42 2017

@author: jacobfisher
"""

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models 
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()
l_stemmer = LancasterStemmer()

tokenizer = RegexpTokenizer(r'\w+')
results = []

with open('NYT_Immigration_edited.txt', 'r') as myfile:
    data = myfile.read().splitlines()
    for line in data:
        results.append(line)
        

#with open('NYT_Immigration.txt') as inputfile:
#    inputfile.replace("'", "")
#    for line in inputfile:
#        results.append(line)
#        
print(results[0:10])
        
# list for tokenized documents in loop
texts = []
for i in results:
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    stopped_tokens = [i for i in tokens if not i in en_stop]
    texts.append(stopped_tokens)
    
print(texts[:10])
    
dictionary = corpora.Dictionary(texts)
print(dictionary)

corpus = [dictionary.doc2bow(tokens) for tokens in texts]
print(corpus[0])

ldamodel = models.ldamodel.LdaModel(corpus, num_topics=7, id2word = dictionary, passes=40)
print(ldamodel.print_topics(num_topics=5, num_words=5))