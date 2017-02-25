
import gensim.models
import json
import nltk

def remove_specialchars(tstr):
    return tstr\
        .replace(u'\\','')\
        .replace(u'\\n','')\
        .replace(u"'",'')\
        .replace(u'"','')

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
    outfile = "results/model.idk"

    sources = {
        'breitbart': 'Data/scraped_articles_breitbart.json',
        'cbsnews': 'Data/scraped_articles_cbsnews.json',
        'cnn': 'Data/scraped_articles_cnn.json',
        'foxnews': 'Data/scraped_articles_foxnews.json',
        'nytimes': 'Data/scraped_articles_nytimes.json',
        }

    for srcname, srcfile in sources.items():
        articles = get_source_data(srcfile)
        src_sent = list()
        for url, art in articles.items():
            story = art['story_content']
            sent_tokens = nltk.sent_tokenize(str(story))
            sent_tokens = [str(remove_specialchars(sent)) for sent in sent_tokens]

            src_sent = src_sent + sent_tokens

        # train model on all sentences from source
        print('{} sentences for {}.'.format(len(src_sent), srcname))
        print("Training model on {}".format(srcname))
        model = gensim.models.Word2Vec(sent_tokens, size=20,workers=6)
        model.save('results/{}.wtvmodel'.format(srcname))

