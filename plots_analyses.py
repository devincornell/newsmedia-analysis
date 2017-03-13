import nltk
import pickle
import matplotlib.pyplot as plt
import seaborn
import pandas as pd

sources = [
    'results/breitbart_wordfreq.pickle',
    'results/cbsnews_wordfreq.pickle',
    'results/cnn_wordfreq.pickle',
    'results/foxnews_wordfreq.pickle',
    'results/nytimes_wordfreq.pickle',
    'results/wapo_wordfreq.pickle',
    'results/huffpo_wordfreq.pickle',
    'results/reuters_wordfreq.pickle',
    'results/usatoday_wordfreq.pickle',
    'results/watimes_wordfreq.pickle',
]

plt.figure(figsize=(20,12))
for i, srcfile in enumerate(sources):
    print('Loading data file {}.'.format(sources))
    with open(srcfile, 'rb') as infile:
        wordfreq = pickle.load(infile)

    plt.subplot(4, 4, i+1)

    s = pd.Series(list(wordfreq.values()), index=list(wordfreq.keys()))
    s.sort_values(ascending=False)[:10].plot(kind='bar')

    plt.title(srcfile)
    plt.suptitle('Word Frequencies')

plt.show()


'''
# Old Code
import nltk
import pickle
import matplotlib.pyplot as plt
import pandas as pd

with open('/Users/jacobfisher/PycharmProjects/newsmedia-analysis/results/breitbart_wordfreq.pickle', 'rb') as infile:
    wf = pickle.load(infile)

s = pd.Series(list(wf.values()), index=list(wf.keys()))
s.sort_values(ascending=False(kind='bar')

plt.show()


'''



