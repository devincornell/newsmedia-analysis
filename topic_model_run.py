# coding: utf-8
from topic_model import *
import json

data = {}
with open('/Users/jacobfisher/PycharmProjects/newsmedia-analysis/Data/postprocessed_data.json', 'r') as infile:
    for k,v in json.load(infile).items():
        data[k] = v

idf_out = nmf_idf_model([v['story_content'] for v in data.values()])
model = idf_out['nmf']
feature_names = idf_out['feature_names']
save_top_words(model,feature_names,n_top_words,'topics2.txt')
