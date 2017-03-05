# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:29:54 2017

@author: jmm
"""
import re, string, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
sys.path.append('/srv/www/django_mona/')
from monatools.nlptools import Wordbag
from difflib import SequenceMatcher
from itertools import combinations

SHORT_PARA_WORDS = 20 # paragraphs with wordcounts <= this value are considered short
SOURCES = ['nytimes','huffingtonpost','washingtonpost','washingtontimes','reuters','time','usatoday','cbsnews','breitbart','cnn','foxnews']
JSON_FILE_NAME_TEMPLATE = 'scraped_articles_{}.json'

def evaluate_heuristics(article_dict):
    '''
    Use heuristics to score various article features    
    '''
    headline = article_dict['headline']
    story_content = article_dict['story_content']
    metadata = article_dict['metadata']
    wordcount_as_scraped = article_dict['wordcount_as_scraped']
    warnings = article_dict['warnings']
    wb = Wordbag(story_content)
    paras = story_content.split('\n\n')
    quality_heuristics = {}

    # warnings 
    quality_heuristics['warning_count'] = len(warnings)
    suspicious_characters = warnings.get('suspicious_characters')
    quality_heuristics['character_warning_type_count'] = len(suspicious_characters.keys()) if suspicious_characters else 0
    quality_heuristics['character_warning_token_count'] = sum([len(v) for v in suspicious_characters.values()]) if suspicious_characters else 0   
    # regex wordcount
    quality_heuristics['wordcount_regex'] = wordcount_as_scraped
    quality_heuristics['wordcount_gdelt'] = metadata['wordcount']
    quality_heuristics['wordcount_difference'] = wordcount_as_scraped - metadata['wordcount']
    quality_heuristics['wordcount_difference_pct'] = 100 * ((wordcount_as_scraped - metadata['wordcount']) / float(metadata['wordcount']))

    # paragraphs
    quality_heuristics['total_paragraphs'] = len(paras)
 
    # words per paragraph
    lengths = [len(re.findall(r'\w+',''.join([c for c in p if c not in string.punctuation]))) for p in paras]
    quality_heuristics['words_per_paragraph'] = sum(lengths) / float(len(lengths))

    # short paragraphs
    quality_heuristics['total_short_paragraphs'] = len([l for l in lengths if l <= SHORT_PARA_WORDS])
    consecutive_short = 0
    last_short = False
    for l in lengths:
        if l <= SHORT_PARA_WORDS:
            if last_short:
                consecutive_short += 1
            else:
                last_short = True
        else:
            last_short = False
    quality_heuristics['consecutive_short_paragraphs'] = consecutive_short

    # duplicate paragraphs
    quality_heuristics['duplicate_paragraphs_strict_count'] = len(paras) - len(set(paras))
    similarity_array = np.array([SequenceMatcher(None, p1, p2).ratio() for p1,p2 in combinations(paras,2)])
    quality_heuristics['duplicate_paragraphs_pairwise_similarity_max'] = similarity_array.max() if similarity_array.any() else 0.0
    quality_heuristics['duplicate_paragraphs_pairwise_similarity_mean'] = similarity_array.mean() if similarity_array.any() else 0.0
    quality_heuristics['duplicate_paragraphs_pairwise_similarity_std'] = similarity_array.std() if similarity_array.any() else 0.0

    # words per sentence / sentences per paragraph
    quality_heuristics['sentences_per_paragraph'] = len(wb.sentences) / float(quality_heuristics['total_paragraphs'])
    quality_heuristics['words_per_sentence'] = quality_heuristics['wordcount_regex'] / float(len(wb.sentences))

    # % caps/max consecutive caps
    quality_heuristics['total_capital_letters'] = len(re.findall(r'[A-Z]',story_content))
    quality_heuristics['pct_capital_letters'] = 100*((quality_heuristics['total_capital_letters'])/float(len(story_content)))
    quality_heuristics['total_all_caps_words'] = len(re.findall(r'[A-Z]{2,}',story_content))
    quality_heuristics['pct_all_caps_words'] = 100*(quality_heuristics['total_all_caps_words'])/float(quality_heuristics['wordcount_regex'])

    # check for JS/CSS/HTML indicators: "{", "}", "()", "$selector", "foo_bar", "<foo></foo>"
    contamination_score = 0
    if '{' in story_content or '}' in story_content:
        contamination_score += 1
    if '()' in story_content:
        contamination_score += 1
    if re.match(r'\$[A-Za-z]+',story_content):
        contamination_score += 1
    if re.match(r'[A-Za-z]+_[A-Za-z]+',story_content):
        contamination_score += 1
    if re.match(r'</?\w+>',story_content):
        contamination_score += 1
    quality_heuristics['source_code_contamination_score'] = contamination_score

    return quality_heuristics

def select_candidates(articles):
    '''
    Select candidate articles from the center of the distributions of heuristic scores
    '''
    article_scores = {}
    selected_articles = {}
    for url,data in articles.items():
        data['quality_heuristics'] = evaluate_heuristics(**data)

    return selected_articles

def postprocess(article_data):
    '''
    Apply some filtering/cleanup
    '''
    # reduce big sequences of newlines
    article_data['story_content'] = re.sub(r'\n{3,}','\n\n',article_data['story_content'])

    # strip leading/trailing whitespaces from lines
    article_data['story_content'] = '\n\n'.join([para.strip() for para in article_data['story_content'].split('\n\n')])
 
    # reduce spaces
    article_data['story_content'] = re.sub(r'[\t\xa0 ]',' ',article_data['story_content'])

    # TODO: strip out urls (and replace with placeholder to account for tweets that are just links??)

    # return an updated dictionary
    return article_data

def load_data(sources=SOURCES):
    import json
    data = {}
    for source in sources:
        with open(JSON_FILE_NAME_TEMPLATE.format(source),'r') as infile:
            data.update(json.load(infile))
    return data

if __name__ == '__main__':
    pass
