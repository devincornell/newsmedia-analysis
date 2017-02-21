
# server.py
from wordtovec import WordToVec
from flask import Flask
import json



app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/getword/<word>')
def getword(word):

    print('shit')
    resp = {'word':word}
    resp['vec'] = list(wtv.get_word(word))
    # show the user profile for that user
    return json.dumps(resp)

# @app.route('/post/<int:post_id>')
# def show_post(post_id):
#     # show the post with the given id, the id is an integer
#     return 'Post %d' % post_id


if __name__ == "__main__":
    fname = "data/GoogleNews-vectors-negative300.bin.gz"
    wtv = WordToVec(fname, testwords=500)
    #print(wtv.df.head())

    print('runnin server now')
    app.run()