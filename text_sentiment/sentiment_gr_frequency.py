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
            bigrams = ngrams(token, 3)
            counter += Counter(bigrams)
        print(filename) 
        return counter

import socket
hostname = socket.gethostname().split('-')
if not hostname[0] == 'circinus':
    exit()
hostname_num = int(hostname[1])
if hostname_num < 10 or hostname_num > 19:
    exit()
padding = 10
num_cpus = 20
file_per_host = 40
pool = mp.Pool(num_cpus)
starting = (hostname_num - padding) * file_per_host + 1
stopping = (hostname_num - padding + 1) * file_per_host + 1
print(starting, stopping)

# uncomment each part before running the script

'''
jobs = ['sentiment_gr_positive/sentiment_gr_positive_{}.tsv'.format(k) for k in range(starting, stopping)]
results = pool.map(get_word_freq, jobs)
top_words_positive = combine_result(results,50)
data_positive=pd.DataFrame()
data_positive['words']=[val[0] for val in top_words_positive]
data_positive['freq']=[val[1] for val in top_words_positive]
data_positive.to_csv('sentiment_gr_positive_unigram.csv', index = False)
'''
#--------------------------------------------
'''
jobs = ['sentiment_gr_negative/sentiment_gr_negative_{}.tsv'.format(k) for k in range(starting,stopping)]
results = pool.map(get_word_freq, jobs)
top_words_negative = combine_result(results,50)
data_negative=pd.DataFrame()
data_negative['words']=[val[0] for val in top_words_negative]
data_negative['freq'] =[val[1] for val in top_words_negative]
data_negative.to_csv('sentiment_gr_negative_unigram.csv', index = False)
'''
#-------------------------------------------------
'''
jobs = ['sentiment_gr_positive/sentiment_gr_positive_{}.tsv'.format(k) for k in range(starting, stopping)]
results = pool.map(get_bigram_freq, jobs)
top_words_positive = combine_result(results,50)
data_positive=pd.DataFrame()
data_positive['words']=[val[0] for val in top_words_positive]
data_positive['freq']=[val[1] for val in top_words_positive]
data_positive.to_csv('sentiment_gr_positive_bigram.csv', index = False)
'''
#-----------------------------------------------------
'''
jobs = ['sentiment_gr_negative/sentiment_gr_negative_{}.tsv'.format(k) for k in range(starting,stopping)]
results = pool.map(get_bigram_freq, jobs)
top_words_negative = combine_result(results,50)
data_negative=pd.DataFrame()
data_negative['words']=[val[0] for val in top_words_negative]
data_negative['freq'] =[val[1] for val in top_words_negative]
data_negative.to_csv('sentiment_gr_negative_bigram.csv', index = False)
'''
#--------------------------------------------------------
jobs = ['sentiment_gr_positive/sentiment_gr_positive_{}.tsv'.format(k) for k in range(starting, stopping)]
results = pool.map(get_trigram_freq, jobs)
top_words_positive = combine_result(results,60)
data_positive=pd.DataFrame()
data_positive['words']=[val[0] for val in top_words_positive]
data_positive['freq']=[val[1] for val in top_words_positive]
data_positive.to_csv('sentiment_gr_positive_trigram.csv', index = False)

#-----------------------------------------------------------
'''
jobs = ['sentiment_gr_negative/sentiment_gr_negative_{}.tsv'.format(k) for k in range(starting,stopping)]
results = pool.map(get_trigram_freq, jobs)
top_words_negative = combine_result(results,60)
data_negative=pd.DataFrame()
data_negative['words']=[val[0] for val in top_words_negative]
data_negative['freq'] =[val[1] for val in top_words_negative]
data_negative.to_csv('sentiment_gr_negative_trigram.csv', index = False)
'''