import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
import matplotlib.pyplot as plt
import multiprocessing as mp
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import word_tokenize
from nltk.util import ngrams
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#% matplotlib inline


# Counting word frequency
def get_word_freq(filename):
    counter = Counter()
    with open(filename, 'r') as f:
        next(f)
        for i, line in enumerate(f):
            text = line.strip().split(' ')
            counter += Counter(text)
    print(filename) 
    return counter

def combine_result(results, n):
    counter = Counter()
    for c in results:
        counter += c
    return counter.most_common(n)
# Counting word frequency for bigram
def get_bigram_freq(filename):
    counter = Counter([])
    with open(filename, 'r') as f:
        next(f)
        for i, line in enumerate(f):
            text = line.strip().split('\t')
            token = nltk.word_tokenize(text[1])
            bigrams = ngrams(token, 2)
            counter += Counter(bigrams)
        print(filename) 
        return counter
def get_trigram_freq(filename):
    counter = Counter([])
    with open(filename, 'r') as f:
        next(f)
        for i, line in enumerate(f):
            text = line.strip().split('\t')
            token = nltk.word_tokenize(text[1])
            trigrams = ngrams(token, 3)
            counter += Counter(trigrams)
        print(filename) 
        return counter
#---------------------------------------------------------
# Using multiprocessing
import socket
hostname = socket.gethostname().split('-')
if not hostname[0] == 'circinus':
    exit()
hostname_num = int(hostname[1])
if hostname_num < 10 or hostname_num > 19:
    exit()
padding = 10
num_cpus = 20
file_per_host = 50
pool = mp.Pool(num_cpus)
starting = (hostname_num - padding) * file_per_host + 1
stopping = (hostname_num - padding + 1) * file_per_host + 1
print(starting, stopping)

# uncomment each part before running the script
'''
jobs = ['sentiment_amazon_positive/sentiment_amazon_positive_{}.tsv'.format(k) for k in range(starting, stopping)]
results = pool.map(get_word_freq, jobs)
top_words_positive = combine_result(results,50)
data_positive=pd.DataFrame()
data_positive['words']=[val[0] for val in top_words_positive]
data_positive['freq']=[val[1] for val in top_words_positive]
data_positive.to_csv('sentiment_amazon_positive_unigram.csv', index = False)
'''
#---------------------------
'''
jobs = ['sentiment_amazon_negative/sentiment_amazon_negative_{}.tsv'.format(k) for k in range(starting,stopping)]
results = pool.map(get_word_freq, jobs)
top_words_negative = combine_result(results,50)
data_negative=pd.DataFrame()
data_negative['words']=[val[0] for val in top_words_negative]
data_negative['freq'] =[val[1] for val in top_words_negative]
data_negative.to_csv('sentiment_amazon_negative_unigram.csv', index = False)
'''
#---------------------------
'''
jobs = ['sentiment_amazon_positive/sentiment_amazon_positive_{}.tsv'.format(k) for k in range(starting, stopping)]
results = pool.map(get_biagram_freq, jobs)
top_words_positive = combine_result(results, 50)
data_positive=pd.DataFrame()
data_positive['words']=[val[0] for val in top_words_positive]
data_positive['freq']=[val[1] for val in top_words_positive]
data_positive.to_csv('sentiment_amazon_positive_biagram.csv', index = False)
'''
#------------------------------
'''
jobs = ['sentiment_amazon_negative/sentiment_amazon_negative_{}.tsv'.format(k) for k in range(starting,stopping)]
results = pool.map(get_biagram_freq, jobs)
top_words_negative = combine_result(results,50)
data_negative=pd.DataFrame()
data_negative['words']=[val[0] for val in top_words_negative]
data_negative['freq'] =[val[1] for val in top_words_negative]
data_negative.to_csv('sentiment_amazon_negative_bigram.csv', index = False)
'''
#------------------------------
# Since the file is too big, save each file to pickle file and count the top 50 frequencies later
'''
jobs = ['sentiment_amazon_positive/sentiment_amazon_positive_{}.tsv'.format(k) for k in range(starting, stopping)]
results = pool.map(get_trigram_freq, jobs)
for i,a in enumerate(results):
    num = i + starting
    with open('a_trigram_positive/a_trigram_positive_' +str(num)+ '.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''
#------------------------------------
jobs = ['sentiment_amazon_negativev/sentiment_amazon_negative_{}.tsv'.format(k) for k in range(starting, stopping)]
results = pool.map(get_trigram_freq, jobs)
for i,a in enumerate(results):
    num = i + starting
    with open('a_trigram_negative/a_trigram_negative_' +str(num)+ '.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)