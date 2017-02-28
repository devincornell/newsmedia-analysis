n_top_words = 25
def save_top_words(model, feature_names, n_top_words, outfile):
    outstring = ''
    for topic_idx, topic in enumerate(model.components_):
        outstring += 'Topic #{}:\n'.format(topic_idx)
        outstring += ' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        outstring += '\n'
    with open(outfile,'w') as outfile:
        outfile.write(outstring)

def nmf_idf_model(samples):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    n_features = 10000
    n_topics = 25
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(samples)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
    return {'tfidf':tfidf,'nmf':nmf,'feature_names':tfidf_feature_names}

