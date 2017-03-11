
# server.py
import gensim
from flask import Flask
import json



app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/getword/<word>')
def getword(word):

    resp = {'word':word}
    resp['vec'] = list(model[word])

    # show the user profile for that user
    return json.dumps(resp)

# @app.route('/post/<int:post_id>')
# def show_post(post_id):
#     # show the post with the given id, the id is an integer
#     return 'Post %d' % post_id


if __name__ == "__main__":
    fname = "data/GoogleNews-vectors-negative300.bin"
    #fname = "data/googlenews.bin"
    model = gensim.models.Word2Vec.load_word2vec_format(fname, binary=True)

    #wtv = WordToVec(fname, testwords=500)
    #print(wtv.df.head())

    print('runnin server now')
    app.run()