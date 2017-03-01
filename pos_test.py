import nltk

src_sent = [['faithfocused', 'piece', 'new', 'york', 'times', 'leftleaning', 'paper', 'gushes', 'religious', 'institutions', 'across', 'country', 'offering', 'sanctuary', 'migrant', 'felons'], ['piece', 'titled', 'houses', 'worship', 'poised', 'serve', 'trumpera', 'immigrant', 'sanctuaries', 'showcases', 'movement', 'known', 'sanctuary', 'deportation', 'churches', 'help', 'shield', 'illegal', 'immigrants', 'federal', 'law', 'enforcement'], ['instance', 'times', 'writeup', 'mentions', 'one', 'church', 'actively', 'helping', 'harbor', 'illegal', 'immigrant', 'felon', 'javier', 'flores', 'garcia', 'mexico', 'facing', 'deportation', 'already', 'deported', 'three', 'times', 'prior'], ['flores', 'took', 'refuge', 'arch', 'street', 'united', 'methodist', 'church', 'according', 'times', 'bypassed', 'federal', 'immigration', 'officials', 'failed', 'arrive', 'scheduled', 'deportation', 'back', 'mexico:', 'federal', 'immigration', 'authorities', 'say', 'mr.', 'flores', 'long', 'history', 'violations:', 'apprehended', 'nine', 'times', '1997', '2002', 'trying', 'cross', 'border'], ['reentered', 'ordered', 'removed', 'judge', '2007'], ['reentered', 'twice', '2014', 'served', 'prison', 'sentences', 'illegal', 'reentry', 'criminal', 'felony', 'conviction'], ['flores', 'told', 'times', 'crime', 'coming', 'back', 'referring', 'number', 'times', 'federal', 'immigration', 'officials', 'deport', 'mexican', 'national'], ['efforts', 'churches', 'shield', 'illegal', 'immigrants', 'federal', 'laws', 'ongoing', 'throughout', 'obama', 'administration', 'trotted', 'mainstream', 'media', 'ever', 'presidentelect', 'donald', 'trump', 'set', 'head', 'washington', 'd.c'], ['trumps', 'immigration', 'policies', 'businesses', 'verify', 'employees', 'legal', 'residents', 'country;', 'foreign', 'guest', 'worker', 'visa', 'programs', 'see', 'crackdown;', 'border', 'wall', 'expected', 'erected', 'along', 'southern', 'border'], ['times', 'heralded', 'idea', 'illegal', 'immigrants', 'longer', 'rely', 'white', 'americans', 'help', 'quoting', 'pastor', 'salvatierra', 'saying', 'different', 'universe']]

# print(src_sent[:10])

src_sent = [nltk.pos_tag(x) for x in src_sent]

for list in src_sent:
    filter(lambda x: x[1] == "NN", list)

print(src_sent[:10])

# src_sent_nouns = []
# for sent in src_sent:
#     nouns = [filter(lambda x: x[1] == "NN", src_sent)]
#     src_sent_nouns.append(nouns)
#
# print(src_sent_nouns[:10])