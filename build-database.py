import semanticanalysis as sa
from datetime import datetime
from articles import *
import json


sources = {
    'breitbart': 'Data/scraped_articles_breitbart.json',
    'cbsnews': 'Data/scraped_articles_cbsnews.json',
    'cnn': 'Data/scraped_articles_cnn.json',
    'foxnews': 'Data/scraped_articles_foxnews.json',
    'nytimes': 'Data/scraped_articles_nytimes.json',
    'wapo': 'Data/scraped_articles_washingtonpost.json',
    'huffpo': 'Data/scraped_articles_huffingtonpost.json',
    'reuters': 'Data/scraped_articles_reuters.json',
    'usatoday': 'Data/scraped_articles_usatoday.json',
    'watimes': 'Data/scraped_articles_washingtontimes.json',
    }



articles = Articles('results/articles.db')
for srcname, fname in sources.items():
    with open(fname,encoding='utf-8', errors='surrogateescape') as f:
        dat = json.load(f)
    for url, dat in dat.items():
        yr = int(str(dat['metadata']['timestamp'])[0:4])
        mo = int(str(dat['metadata']['timestamp'])[4:6])
        day = int(str(dat['metadata']['timestamp'])[6:8])
        hr = int(str(dat['metadata']['timestamp'])[8:10])
        dt = datetime(year=yr, month=mo, day=day, hour=hr)
        text = dat['story_content'].encode('utf-8').decode()
        
        articles.add({
            'source':dat['metadata']['source'],
            'date': dt.timestamp(),
            'datestr': dt.strftime("%Y-%m-%d %H"),
            'url': url,
            'meta': dat['metadata'],
            'text': text,
            'headline': dat['headline'],
        })
        
        
        
        
print(articles.getdf(limit=1))
print(articles)