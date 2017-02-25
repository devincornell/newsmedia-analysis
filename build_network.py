
import json
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import networkx as nx
from multiprocessing import Pool

parallel = True

flist = [
	'Data/scraped_articles_breitbart.json',
	'Data/scraped_articles_cbsnews.json',
	'Data/scraped_articles_cnn.json',
	'Data/scraped_articles_foxnews.json',
	'Data/scraped_articles_nytimes.json',
	]


# map function to get cofrequency given a paragraph of text
def get_cofreq(allp):
	tokenizer = RegexpTokenizer(r'\w+')
	en_stop = set(stopwords.words('english'))

	# tokenize paragraphs
	pars = list()
	for i in range(len(allp)):
		allp[i] = allp[i]\
			.replace('\u2019','')\
			.replace('\u2014','')\
			.replace('\u201c','')\
			.replace('\u201d','')\
			.replace('\u2018','')

		# paragraphs as a set of tokens
		tokens = tokenizer.tokenize(allp[i].lower())
		tokens = set([i for i in tokens if not i in en_stop])
	pars.append(tokens)

	# get co-occurrence frequencies
	cofreq = {(a,b):0 for p in pars for a in p for b in p}
	for p in pars:
		for a in p:
			for b in p:
				cofreq[(a,b)] += 1
	
	# This will be done in build graph step...
	# remove zeros and reciprocols
	#cofreq = {k:v for k,v in cofreq.items() if v > 0}
	#okeys = set(cofreq.keys())
	#for k in okeys:
	#	if (k[1],k[0]) in cofreq.keys():
	#		cofreq[k] += cofreq[(k[1],k[0])]
	#		del cofreq[(k[1],k[0])]
	return cofreq


if __name__ == '__main__':
	# loop through each article of each source
	articles = list() # list of all paragraphs
	for fname in flist:
		print('Loading data file {}.'.format(fname))
		with open(fname, 'r') as f:
			dat = json.load(f)

		for url, dat in dat.items():
			# get metadata
			yr = str(dat['metadata']['timestamp'])[0:4]
			mo = str(dat['metadata']['timestamp'])[4:6]
			
			# split into paragraphs
			pars = dat['story_content'].split('\u00a0')
			articles.append(pars)

	articles = articles[0:100]

	# map paragraph strings to cofrequency counts
	if parallel:
		print('Applying parallel map..')
		p = Pool(10)
		art_cofreq = list(p.map(get_cofreq, articles))
	else:
		print('Applying map..')
		art_cofreq = list(map(get_cofreq, articles))

	# combine cofreq from all articles
	print('Combining cofreq counts from articles..')
	cofreq = dict()
	for art in art_cofreq:
		for k in art.keys():
			if k in cofreq.keys():
				cofreq[k] += art[k]
			else:
				cofreq[k] = art[k]

	# construct graph
	print('Constructing Graph..')
	G = nx.Graph()
	for k in cofreq.keys():
		#if (k[0],k[1]) not in G.edges():
		try:
			G.edge[k[0]][k[1]]
		except:
			G.add_edge(k[0],k[1],count=0)
		G.edge[k[0]][k[1]]['count'] += 1

	# apply edge attribute weight (inverse of count)
	weights = {(e[0],e[1]):1/e[2]['count'] for e in G.edges(data=True)}
	nx.set_edge_attributes(G,'weight', weights)

	# save file
	nx.write_gexf(G,'test.gexf')
	print('Saved network .gexf file.')



