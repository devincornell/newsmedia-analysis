


def build_semanticnetwork(model=None,text=None,verbose=True):
    ''' This function will build a semantic network
    using word2vec for a set of text blocks. The text 
    blocks 

    text: format=[{'word':'attr'}] If provided, will first
    build a word2vec model from words then apply them to 
    nodes.
    '''

    if len(text) > 0 and 

def build_wtvmodel(text=None, verbose=True):

    print('{} sentences for {}.'.format(len(src_par), srcname))
    print("Training model on {}".format(srcname))
    model = gensim.models.Word2Vec(src_par, size=num_dim,workers=6)
    print('{} contains {} words.'.format(srcname,len(set(model.vocab.keys()))))


def save_model(model=None, wordattr=None, verbose=False):
    if verbose: print('Saving model..')

def open_model(model=None, model_file=None, wordattr=None, wordattr_file=None, verbose=False):
    
    if model_file is not None and model is not None:
        # save model and word frequency count
        if verbose: print('Saving {} and {}.'format(srcname, srcname))
        model.save(model_file)
        
    if wordattr_file is not None and wordattr is not None:
        with open(wordattr_file,'wb') as f:
            pickle.dump(freq_dist, f)
        if verbose: print()
