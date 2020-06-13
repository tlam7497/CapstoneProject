import numpy as np
#import vaex, numpy as np
from matplotlib.font_manager import FontProperties
import operator
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
#sns.set_style("darkgrid",{"axes.axisbelow" : False })
import string
import pickle
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.test.utils import datapath
import logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


print('--------------------1')
data_words = []
asin = []
with open('LDA_textreviews_goodreads.tsv','r') as f:
    next(f)
    for i, line in enumerate(f):
        #if i == 4:
        #   break
        text = line.strip().split('\t')
        asin.append(text[0])
        data_words.append(text[1])
for i in range(len(data_words)):
    data_words[i] = data_words[i].split()

print('--------------------2')
# Build the bigram 
bigram = gensim.models.Phrases(data_words, min_count = 5, threshold = 100) # higher threshold fewer phrases.
# Faster way to get a sentence clubbed as a bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
print('--------------------3')
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
data_words_bigrams = make_bigrams(data_words)
print('--------------------4')
# Create Dictionary
id2word = corpora.Dictionary(data_words_bigrams)

# Create Corpus
texts = data_words_bigrams
print('--------------------5')
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

print('--------------------6')


'''
print('-------------------3')
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        #model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word, workers = 2, random_seed = 256)
        model = gensim.models.ldamulticore.LdaMulticore(corpus= corpus, id2word=id2word, num_topics= num_topics, workers=6, random_state = 0)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

print('-------------------4')
# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_words_bigrams, start=5, limit=45, step=5)

print('--------------------5')
limit = 45; start = 5; step = 5;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Number Topics")
plt.ylabel("Coherence score")
plt.title('Optimal Number of LDA Topics with Coherence score (Goodreads)')
#plt.legend(("coherence_values"), loc='best')
plt.savefig('LDA_goodreads_optimal_topic_step5.png')
topic = {}
i = 0
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    topic[m] = cv, i
    highest = max(topic.items(), key=operator.itemgetter(1))[1][1]
    coherence = max(topic.items(), key=operator.itemgetter(1))[1][0]
    maxNum  =  max(topic.items(), key=operator.itemgetter(1))[0]
    i+=1
print('--------------------6')
optimal_model = model_list[highest]
model_topics = optimal_model.show_topics(formatted=False)
print(optimal_model.print_topics(num_words=10))
ldamodel=optimal_model
corpus=corpus

print('--------------------7')
num_topic = maxNum
with open ('gr_LDA_wmallet_step5_metadata.csv', 'w') as w: 
    for i,row in enumerate(ldamodel[corpus]):
        w.write(asin[i])
        w.write(',')
        #print(i,row)
        for j,(topic_num,prop_topic) in enumerate(row):
            w.write(str(prop_topic))
            w.write(',')
        
        w.write('\b\n')
print('\nNumer of topic : ', num_topic )
print('\nCoherence Score: ', np.round(coherence,4))
'''
# RUNNING MODEL WITH MALLET
mallet_path = 'mallet-2.0.8/bin/mallet'
model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word, workers = 2, random_seed = 0)

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=model, texts=data_words_bigrams, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
print('--------------------3')

#optimal_model = model
#model_topics = optimal_model.show_topics(formatted=False)
ldamodel=model

with open ('gr_LDA_wmallet_metadata.csv', 'w') as w: 
    for i,row in enumerate(ldamodel[corpus]):
        w.write(asin[i])
        w.write(',')
        #print(i,row)
        for j,(topic_num,prop_topic) in enumerate(row):
            w.write(str(prop_topic))
            w.write(',')
        
        w.write('\b\n')