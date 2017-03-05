
import gensim.models
import json
from nltk import pos_tag
import nltk
import nltk.corpus
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
import matplotlib.pyplot as pp
import pprint
import re

pp.style.use('ggplot')
ptprint = (pprint.PrettyPrinter(indent=3))

def remove_specialchars(paragraph,stopwords,specialchars):
    newpar = [w for w in paragraph if w not in stopwords]
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
    results_folder = 'results/'
    num_dim = 50
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

        # remove special characters and stopwords
        src_par = [remove_specialchars(par,stopwords,specialchars) for par in src_par]

        # convert to lower case
        src_pars = [w.lower() for par in src_par for w in par]

        # # calculate frequency information for each word
        freq_dist = nltk.FreqDist([w for s in src_par for w in s])

        # train model on all sentences from source
        print('{} sentences for {}.'.format(len(src_par), srcname))
        print("Training model on {}".format(srcname))
        model = gensim.models.Word2Vec(src_par, size=num_dim,workers=6)
        print('{} contains {} words.'.format(srcname,len(set(model.vocab.keys()))))
        
        # save model and word frequency count
        print('Saving {}.wtvmodel and {}_wordfreq.pickle'.format(srcname, srcname))
        model.save('{}{}.wtvmodel'.format(results_folder,srcname))
        with open(results_folder + srcname + '_wordfreq.pickle','wb') as f:
            pickle.dump(freq_dist, f)
        print()
        #
