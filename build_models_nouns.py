
import gensim.models
import json
from nltk import pos_tag
import nltk
import nltk.corpus
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
import matplotlib.pyplot as pp
import pprint

pp.style.use('ggplot')
ptprint = (pprint.PrettyPrinter(indent=3))

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
    num_dim = 50

    sources = {
        'breitbart': 'Data/scraped_articles_breitbart.json',
        # 'cbsnews': 'Data/scraped_articles_cbsnews.json',
        # 'cnn': 'Data/scraped_articles_cnn.json',
        # 'foxnews': 'Data/scraped_articles_foxnews.json',
        # 'nytimes': 'Data/scraped_articles_nytimes.json',
        # 'wapo': 'Data/scraped_articles_washingtonpost.json',
        # 'huffpo': 'Data/scraped_articles_huffingtonpost.json',
        # 'reuters': 'Data/scraped_articles_reuters.json',
        # 'usatoday': 'Data/scraped_articles_usatoday.json',
        # 'watimes': 'Data/scraped_articles_washingtontimes.json',
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
        print("Making lists of lowercase words")
        src_sent = [list(map(lambda x: x.lower(),sent.split())) for sent in src_sent]

        # remove a period at the end of every sentence
        for i in range(len(src_sent)):
            if len(src_sent[i]) > 0 and len(src_sent[i][-1]) > 0 and src_sent[i][-1][-1] == '.':
                src_sent[i][-1] = src_sent[i][-1][:-1]

        # apply ascii encodings (by ommitting non-ascii chars)
        src_sent = [list(map(lambda x: x.encode('ascii',errors='ignore').decode(), sent)) for sent in src_sent]

        # remove stopwords from sentences
        print("Removing stopwords")
        src_sent = [list(filter(lambda x: x not in stopwords, sent)) for sent in src_sent]

        print(src_sent[:10])

        # POS Tagging
        print("Tagging parts of speech")
        src_sent_tagged = []
        for sent in src_sent[:10]:
            if sent is None or sent == [] or sent == ():
                src_sent.remove(sent)
                continue
            else:
                try:
                    n = nltk.pos_tag(sent)
                    src_sent_tagged.append(n)
                except IndexError:
                    continue

        print(src_sent_tagged[:10])

        #create list of nouns
        print("Creating a list of nouns")

        src_nouns = []
        for sent in src_sent_tagged:
            if sent is None or sent == [] or sent == ():
                continue
            else:
                src_nouns += ([list(filter(lambda x: x[-1] == 'NN', sent))])

        ptprint.pprint(src_nouns[:100])

        #get rid of the "NN"s from the list
        print("Removing everything but nouns")
        src_nouns_2 = []
        for sent in src_nouns:
            if sent is None or sent == [] or sent == ():
                continue
            else:
                for sent in src_nouns:
                    n = [x[0] for x in sent]
                    src_nouns_2.append(n)

        src_nouns = src_nouns_2
        ptprint.pprint(src_nouns)

        # #create a giant string of all of the nouns (for use later)
        # nounstring = ' '.join(str(word) for sent in src_nouns for word in sent)
        # nounstring.replace('?', '')
        # nounstring.replace('.', '')
        # nounstring.replace('it?', '')
        #
        # print(nounstring)
        #
        # # # lemmatize words using wordnet corpus
        # # lmtzr = WordNetLemmatizer()
        # # src_sent = [list(map(lambda x:lmtzr.lemmatize(x),sent)) for sent in src_sent]
        #
        #
        # # calculate frequency information for each word
        # freq_dist = nltk.FreqDist([w for s in src_nouns for w in s])

        # # train model on all sentences from source
        # print('{} sentences for {}.'.format(len(src_nouns), srcname))
        # print("Training model on {}".format(srcname))
        # model = gensim.models.Word2Vec(src_nouns, size=num_dim,workers=6)
        # print('{} contains {} words.'.format(srcname,len(set(model.vocab.keys()))))
        #
        # # save model and word frequency count
        # print('Saving {}.wtvmodel and {}_wordfreq.pickle'.format(srcname, srcname))
        # model.save('{}{}.wtvmodel'.format(results_folder,srcname))
        # with open(results_folder + srcname + '_wordfreq.pickle','wb') as f:
        #     pickle.dump(freq_dist, f)
        # print()

