import nltk

src_sents = [['faithfocused', 'piece', 'new', 'york', 'times', 'leftleaning', 'paper', 'gushes', 'religious', 'institutions', 'across', 'country', 'offering', 'sanctuary', 'migrant', 'felons'], ['piece', 'titled', 'houses', 'worship', 'poised', 'serve', 'trumpera', 'immigrant', 'sanctuaries', 'showcases', 'movement', 'known', 'sanctuary', 'deportation', 'churches', 'help', 'shield', 'illegal', 'immigrants', 'federal', 'law', 'enforcement'], ['instance', 'times', 'writeup', 'mentions', 'one', 'church', 'actively', 'helping', 'harbor', 'illegal', 'immigrant', 'felon', 'javier', 'flores', 'garcia', 'mexico', 'facing', 'deportation', 'already', 'deported', 'three', 'times', 'prior'], ['flores', 'took', 'refuge', 'arch', 'street', 'united', 'methodist', 'church', 'according', 'times', 'bypassed', 'federal', 'immigration', 'officials', 'failed', 'arrive', 'scheduled', 'deportation', 'back', 'mexico:', 'federal', 'immigration', 'authorities', 'say', 'mr.', 'flores', 'long', 'history', 'violations:', 'apprehended', 'nine', 'times', '1997', '2002', 'trying', 'cross', 'border'], ['reentered', 'ordered', 'removed', 'judge', '2007'], ['reentered', 'twice', '2014', 'served', 'prison', 'sentences', 'illegal', 'reentry', 'criminal', 'felony', 'conviction'], ['flores', 'told', 'times', 'crime', 'coming', 'back', 'referring', 'number', 'times', 'federal', 'immigration', 'officials', 'deport', 'mexican', 'national'], ['efforts', 'churches', 'shield', 'illegal', 'immigrants', 'federal', 'laws', 'ongoing', 'throughout', 'obama', 'administration', 'trotted', 'mainstream', 'media', 'ever', 'presidentelect', 'donald', 'trump', 'set', 'head', 'washington', 'd.c'], ['trumps', 'immigration', 'policies', 'businesses', 'verify', 'employees', 'legal', 'residents', 'country;', 'foreign', 'guest', 'worker', 'visa', 'programs', 'see', 'crackdown;', 'border', 'wall', 'expected', 'erected', 'along', 'southern', 'border'], ['times', 'heralded', 'idea', 'illegal', 'immigrants', 'longer', 'rely', 'white', 'americans', 'help', 'quoting', 'pastor', 'salvatierra', 'saying', 'different', 'universe']]
src_pars = [['ROME -- The United States and Russia are studying new ways to break a months-long diplomatic deadlock over how to stop the fighting in the Syrian city of Aleppo, U.S. Secretary of State John Kerry said Friday.\xa0', 'Kerry said the “ideas” will be tested in follow-up discussions between American and Russian diplomats next week. Kerry would only describe those fresh approaches as designed to lead to talks between Syria’s government and rebels, a goal that has remained elusive since early 2014, while stressing that the U.S. and Russia won’t wait for Donald Trump’s presidency to begin on Jan. 20.', 'But given the repeated failures of the former Cold War foes to halt Syria’s 5 ½-year civil war, it is unclear how much hope the new effort holds. “We have exchanged a set of ideas, which there will be a meeting on early next week in Geneva, and we have to wait and see whether those ideas have any legs to them,” Kerry said after meeting Russian Foreign Minister Sergey Lavrov in Rome. “I will say that both sides understand the importance of trying to continue the diplomacy and trying to see if something can be done. Nobody is waiting for the next administration. We both feel there is urgency.”\xa0', 'Kerry said he will gauge progress with Lavrov when they meet again on the sidelines of a European security conference in Hamburg, Germany, on Wednesday. “Nobody is resigned to the violence,” the American said at a news conference later in the day. While the talks were going on, Syria showed off its recent gains in Aleppo, once the country’s largest city and commercial center. State media reported Friday from areas captured this week in a Russian-backed ground offensive, airing reports of roads being restored, debris removed and civilians resettled. The U.N. aid agency said an estimated 31,500 people have been displaced as a result of the recent fighting, which takes Syrian President Bashar Assad’s government closer to capturing the whole city and completing what would be perhaps a devastating blow to U.S.-backed rebel forces. The war has killed as many as half a million people since 2011, contributed to Europe’s worst refugee crisis since World War II and allowed the Islamic State group to emerge as a global terror threat.\xa0', 'Friday’s diplomatic discussions took place in a hotel several stories above an Italian-hosted conference on the Mediterranean region, and Russia’s Lavrov emphasized that his country won’t allow Syria to follow the example of lawless Libya after NATO’s 2011 intervention that helped topple dictator Muammar Qaddafi. That country now is experiencing perhaps its worst violence in two years as rival militias and extremist groups such as IS continue to vie for power. While Washington has accused Moscow of war crimes and crimes against humanity in Syria, Lavrov blamed both the U.S. and United Nations for the current situation. He lamented that the U.S. has been unable to fulfill its commitment under several past cease-fire plans to separate the so-called “moderate” opposition groups from the al-Qaeda-linked fighters that Russia says it is targeting. And he questioned why the U.N. isn’t restarting peace talks or rushing aid to areas of Syria in need, something the global body has been extremely reticent to do since a September convoy was hit by an airstrike. The U.S. has blamed Russia for that attack, a charge Moscow denies. “The time is ripe for compromise,” Lavrov said. Both diplomats met Friday with the U.N.’s envoy for Syria, Staffan de Mistura. As de Mistura began his meeting with Kerry, reporters could hear the peace mediator telling the U.S. secretary of state, “Anything but stalemate.”']]
tokens = []
for sent in src_pars:
    n = [nltk.word_tokenize(word) for word in sent]
    tokens.append(n)

print(tokens[:10])

# src_sents = [nltk.pos_tag(x) for x in src_sents]
#
# src_nouns = []
# for sent in src_sents:
#     n = [x for x in sent if x[-1] == 'NN']
#     src_nouns.append(n)
#
# src_nouns_2 = []
# for sent in src_nouns:
#     n = [x[0] for x in sent]
#     src_nouns_2.append(n)
#
# print(src_nouns_2)
#
# stringnouns = ' '.join(str(word) for sent in src_nouns_2 for word in sent)
#
# print(stringnouns)

# src_sent_nouns = []
# for sent in src_sents:
#     nouns = [filter(lambda x: x[1] == "NN", src_sent)]
#     src_sent_nouns.append(nouns)
#
# print(src_sent_nouns[:10])