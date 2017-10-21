
import gensim.models
import json
from nltk import pos_tag
import nltk
import nltk.corpus
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
import pprint
import re

punctlist = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '``']

def remove_specialchars(paragraph,stopwords,specialchars):
    newpar = [w for w in paragraph if w not in stopwords]
    newpar = [w for w in newpar if w not in punctlist]
    return newpar


def get_source_data(fname,verbose=False):
    if verbose: print('Loading data file {}.'.format(fname))
    with open(fname,encoding='utf-8', errors='surrogateescape') as f:
        filestr = f.read()
        dat = json.loads(filestr)

    for url, data in dat.items():
        # get metadata
        data['metadata']['yr'] = str(data['metadata']['timestamp'])[0:4]
        data['metadata']['mo'] = str(data['metadata']['timestamp'])[4:6]
        data['story_content'] = data['story_content'].encode('utf-8')

    return dat


if __name__ == "__main__":

    # script params
    results_folder = 'results/wtvmodels/'
    num_dim = 100
    specialchars = ["'",'"',',','.','&']

    sources = {
        'breitbart': 'Data/scraped_articles_breitbart.json',
        'cbsnews': 'Data/scraped_articles_cbsnews.json',
        'cnn': 'Data/scraped_articles_cnn.json',
        'foxnews': 'Data/scraped_articles_foxnews.json',
        'nytimes': 'Data/scraped_articles_nytimes.json',
        'wapo': 'Data/scraped_articles_washingtonpost.json',
        'huffpo': 'Data/scraped_articles_huffingtonpost.json',
        'reuters': 'Data/scraped_articles_reuters.json',
        'usatoday': 'Data/scraped_articles_usatoday.json',
        'watimes': 'Data/scraped_articles_washingtontimes.json',
        }

    stopwords = nltk.corpus.stopwords.words('english')
    all_paragraphs = list()
    for srcname, srcfile in sources.items():
        articles = get_source_data(srcfile)
        src_par = list()
        for url, art in articles.items():
            story = art['story_content'].decode('ascii', errors='ignore')
            pars = story.split('\n\n')

            for p in pars:
                src_par.append(nltk.word_tokenize(p))

        # remove special characters
        src_par = [[w for w in par if w not in punctlist] for par in src_par]

        # convert to lower case
        src_par = [[w.lower() for w in par] for par in src_par]

        # remove stopwords
        src_par = [[w for w in par if w not in stopwords and w.isalnum()] for par in src_par]

        # train model on all sentences from source
        print('{} sentences for {}.'.format(len(src_par), srcname))
        print("Training model on {}".format(srcname))
        model = gensim.models.Word2Vec(src_par, size=num_dim,workers=8, min_count=3)
        print('{} contains {} unique words.'.format(srcname,len(model.wv.vocab)))

        # save model and word frequency count
        print('Saving {}_pars.wtvmodel'.format(srcname))
        model.save('{}{}_pars.wtvmodel'.format(results_folder,srcname))
        print()
