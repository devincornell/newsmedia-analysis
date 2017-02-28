#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 21:31:02 2017

@author: jacobfisher
"""
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models 

tokenizer = RegexpTokenizer(r'\w+')

doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
doc_e = "Health professionals say that brocolli is good for your health."

doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

doc_set_lower = [str.lower(i) for i in doc_set] 
tokens = [tokenizer.tokenize(i) for i in doc_set_lower]

en_stop = get_stop_words('en')

p_stemmer = PorterStemmer()
stopped_stemmed_tokens = []
for i in tokens:
    doc_tokens = [word for word in i if not word in en_stop]
    stem_tokens = [p_stemmer.stem(i) for i in doc_tokens]
    stopped_stemmed_tokens.append(stem_tokens)

#print(stopped_stemmed_tokens)

dictionary = corpora.Dictionary(stopped_stemmed_tokens)
print(dictionary)

corpus = [dictionary.doc2bow(tokens) for tokens in stopped_stemmed_tokens]
print(corpus[0])

ldamodel = models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=30)
print(ldamodel.print_topics(num_topics=3, num_words=3))