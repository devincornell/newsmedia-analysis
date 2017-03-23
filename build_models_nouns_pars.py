
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

punctlist = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '``']

pp.style.use('ggplot')
ptprint = (pprint.PrettyPrinter(indent=3))

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

        # remove special characters
        src_par = [[w for w in par if w not in punctlist] for par in src_par]

        # POS Tagging
        print("Tagging parts of speech")

        src_par_tagged = []

        for par in src_par:
            if par is None or par == [] or par == ():
                continue
            else:
                try:
                    n = nltk.pos_tag(par)
                    src_par_tagged.append(n)
                except IndexError:
                    continue

        src_par_tagged = [par for par in src_par_tagged if par]

        # create list of nouns
        print("Creating a list of nouns")

        allowed_pos = ['NN', 'NNP', 'NNS']

        src_nouns = [[n for n in par if n and n[-1] in allowed_pos] for par in src_par_tagged]

        # get rid of the "NN"s from the list
        print("Removing everything but nouns")

        src_nouns_2 = [[n[0] for n in par] for par in src_nouns]

        src_nouns = src_nouns_2

        # convert to lower case
        src_par = [[w.lower() for w in par] for par in src_par]

        # remove stopwords
        print('removing stopwords')
        src_par = [[w for w in par if w not in stopwords and w.isalnum()] for par in src_par]

        # # calculate frequency information for each word
        print('calculating frequency distribution for {}'.format(srcname))
        freq_dist = nltk.FreqDist([w for s in src_par for w in s])

        # train model on all sentences from source
        print('{} sentences for {}.'.format(len(src_par), srcname))
        print("Training model on {}".format(srcname))
        model = gensim.models.Word2Vec(src_par, size=num_dim,workers=6)
        print('{} contains {} words.'.format(srcname,len(set(model.vocab.keys()))))

        # save model and word frequency count
        print('Saving {}_nouns.wtvmodel and {}_nounfreq.pickle'.format(srcname, srcname))
        model.save('{}{}_nouns.wtvmodel'.format(results_folder,srcname))
        with open(results_folder + srcname + '_nounfreq.pickle','wb') as f:
            pickle.dump(freq_dist, f)
        print("-" * 20)
