
import gensim.models
import json
import nltk
import nltk.corpus
from nltk.stem.wordnet import WordNetLemmatizer
import pickle

def remove_specialchars(tstr):
    return tstr\
        .replace(u'\\','')\
        .replace(u"'",'')\
        .replace(u'"','')\
        .replace(u'-','')\
        .replace(u',','')\


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

    sources = {
        'breitbart': 'Data/scraped_articles_breitbart.json',
        'cbsnews': 'Data/scraped_articles_cbsnews.json',
        'cnn': 'Data/scraped_articles_cnn.json',
        'foxnews': 'Data/scraped_articles_foxnews.json',
        'nytimes': 'Data/scraped_articles_nytimes.json',
        }
    
    stopwords = nltk.corpus.stopwords.words('english')
    for srcname, srcfile in sources.items():
        articles = get_source_data(srcfile)
        src_sent = list()
        for url, art in articles.items():
            story = art['story_content'].decode('utf-8', errors='ignore')
            sent_tokens = nltk.sent_tokenize(story)
            sent_tokens = [remove_specialchars(sent) for sent in sent_tokens]

            src_sent = src_sent + sent_tokens

        # break each sentence into a list of lower case words without the '.' character
        src_sent = [list(map(lambda x: x.lower(),sent.replace('.','').split())) for sent in src_sent]

        # apply ascii encodings (by ommitting non-ascii chars)
        src_sent = [list(map(lambda x: x.encode('ascii',errors='ignore').decode(), sent)) for sent in src_sent]
        
        # remove stopwords from sentences
        src_sent = [list(filter(lambda x: x not in stopwords, sent)) for sent in src_sent]

        # lemmatize words using wordnet corpus
        lmtzr = WordNetLemmatizer()
        src_sent = [list(map(lambda x:lmtzr.lemmatize(x),sent)) for sent in src_sent]

        # calculate frequency information for each word
        freq_dist = nltk.FreqDist([w for s in src_sent for w in s])

        # train model on all sentences from source
        print('{} sentences for {}.'.format(len(src_sent), srcname))
        print("Training model on {}".format(srcname))
        model = gensim.models.Word2Vec(src_sent, size=20,workers=6)
        print('{} contains {} words.'.format(srcname,len(set(model.vocab.keys()))))

        # save model and word frequency count
        print('Saving {}.wtvmodel and {}_wordfreq.pickle'.format(srcname, srcname))
        model.save('{}{}.wtvmodel'.format(results_folder,srcname))
        with open(results_folder + srcname + '_wordfreq.pickle','wb') as f:
            pickle.dump(freq_dist, f)
        print()

