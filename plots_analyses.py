import nltk
import pickle
import matplotlib.pyplot as plt
import pandas as pd

with open('/Users/jacobfisher/PycharmProjects/newsmedia-analysis/results/breitbart_wordfreq.pickle', 'rb') as infile:
    wf = pickle.load(infile)

s = pd.Series(list(wf.values()), index=list(wf.keys()))
s.sort_values(ascending=False)[:100].plot(kind='bar')

plt.show()



