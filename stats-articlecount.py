import json
import string
from IPython import display

newdata = {}
with open ('Data/postprocessed_data.json', 'r') as infile:
    for k,v in json.load(infile).items():
        newdata[k] = v

sourcenames = ['nytimes','breitbart','foxnews','washingtonpost','cnn','cbsnews']
data = {}
for source in sourcenames:
    with open('Data/scraped_articles_{}.json'.format(source), 'r') as infile:
        for k,v in json.load(infile).items():
            data[k] = v

nytdata = {}
with open('Data/scraped_articles_nytimes.json', 'r') as infile:
    for k,v in json.load(infile).items():
         nytdata[k] = v

breitdata = {}
with open('Data/scraped_articles_breitbart.json', 'r') as infile:
    for k, v in json.load(infile).items():
        breitdata[k] = v

wapodata = {}
with open('Data/scraped_articles_washingtonpost.json', 'r') as infile:
    for k, v in json.load(infile).items():
        wapodata[k] = v

foxdata = {}
with open('Data/scraped_articles_foxnews.json', 'r') as infile:
    for k, v in json.load(infile).items():
        foxdata[k] = v

cnndata = {}
with open('Data/scraped_articles_cnn.json', 'r') as infile:
    for k, v in json.load(infile).items():
        cnndata[k] = v

cbsdata = {}
with open('Data/scraped_articles_cbsnews.json', 'r') as infile:
    for k, v in json.load(infile).items():
        cbsdata[k] = v

strdata = ""
for v in data.values():
    strdata += v['story_content']

strdata.lower()
for ch in string.punctuation:
    strdata = strdata.replace(ch, "")

print("There are in total", len(strdata.split()), "words in the dataset")

print("In total there are", len(newdata), "articles")
print("There are", len(nytdata), "articles from the NYT")
print("There are", len(breitdata), "articles from Breitbart")
print("There are", len(wapodata), "articles from WaPo")
print("There are", len(foxdata), "articles from Fox")
print("There are", len(cnndata), "articles from CNN")
print("There are", len(cbsdata), "articles from CBS")
