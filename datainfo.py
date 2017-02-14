
import json
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import networkx as nx
from multiprocessing import Pool


en_stop = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')

with open('Data/scraped_articles_cbsnews.json', 'r') as f:
	dat = json.load(f)

allp = list()
for url, dat in dat.items():
	yr = str(dat['metadata']['timestamp'])[0:4]
	mo = str(dat['metadata']['timestamp'])[4:6]
	pars = dat['story_content'].split('\u00a0')
	
	#print(len(pars))
	allp = allp + pars

	#exit()

#allp = allp


pars = []
for i in range(len(allp)):

	allp[i] = allp[i]\
		.replace('\u2019','')\
		.replace('\u2014','')\
		.replace('\u201c','')\
		.replace('\u201d','')\
		.replace('\u2018','')
	
	tokens = tokenizer.tokenize(allp[i].lower())
	tokens = set([i for i in tokens if not i in en_stop])
	#print(len(tokens))
	pars.append(tokens)
	
	#print('=====')
	#print(allp[i])

G = nx.Graph()
for p in pars:
	for a in p:
		for b in p:
			try:
				G.edge[a][b]
			except:
				G.add_edge(a,b,weight=0)
			G.edge[a][b]['weight'] += 1

for e in G.edges()


nx.write_gexf(G,'test.gexf')
