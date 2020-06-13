import numpy as np 
import string
import pickle
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

print('--------------------1')
data_words = []
asin = []
with open('LDA_textreviews_am_gr.tsv','r') as f:
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
with open ('LDA_data_words_bigrams.pickle', 'wb') as fp:
    pickle.dump(data_words_bigrams,fp)
print('--------------------4')
# Create Dictionary
id2word = corpora.Dictionary(data_words_bigrams)

# Create Corpus
texts = data_words_bigrams
print('--------------------5')
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

print('--------------------6')

    
with open ('LDA_corpus_bigrams.pickle', 'wb') as fp:
    pickle.dump(corpus, fp)
