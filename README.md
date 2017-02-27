# newsmedia-analysis
This is the IGERT Project for Winter '17 quarter: word2vec of GDELT articles compiled by the media neuroscience lab. The project attempts to capture semantic relational differences of concepts between news sources.

There are two main files in the current pipeline (listed in order):

1. build_models.py
2. build_networks.py

Full descriptions of each can be found below. You might try running them with current parameters before diving into the full descriptions.

=====
1. build_models.py: This file accesses the original data .json files and computes a word2vec projection (parameterized by num_dim in that script) for each news source corpus. This analysis looks at the sentence as the atomic unit and does not distinguish between articles for the projection. It performs this analysis using the following pre-processing steps:

    a) Remove special characters by encoding as ascii with errors ignored and then decoding as a normal string.
    b) The articles are split into sentences using the nltk.sent_tokenize() function.
    c) The following punctuation symbols are removed: ,-'"\
    d) The words are converted to lower case.
    e) Stop words are removed using nltk.corpus.stopwords.words('english')
    f) Stemming is performed to convert each word to its base word using from nltk.stem.wordnet import WordNetLemmatizer (uses WordNet corpus)

Then word frequencies are taken using nltk.FreqDist(), and a word2vec projection is generated using gensim.models.Word2Vec(). The output files are {srcname}.wtvmodel for the gensim word2vec models and {srcname}_wordfreq.pickle for the nltk FreqDist files. These files are expected to be consumed by build_networks.py

=====
2. build_networks.py: This script consumes the .wtvmodel and _wordfreq.pickle files in the results/ folder to build relational networks. The intent is to produce networks managable in size so that they can be viewed. Its pipeline contains the following steps:

    a) open each model and word frequency count
    b) calculate the n most frequenly used words


==== TODO ===== 

build_models.py:

	a) set to use new source

sparsify_networks.py:

	a) move into module


