import numpy as np
import pandas as pd
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import multiprocessing as mp
import glob


'''
filename = glob.glob("a_unigram_positive/*.pickle")
counter = Counter()
for i,file in enumerate(filename):
    print(i)
    with open(file, 'rb') as handle:
        b = pickle.load(handle)
        counter += b
top_words_positive = counter.most_common(50)
data_positive=pd.DataFrame()
data_positive['words']=[val[0] for val in top_words_positive]
data_positive['freq']=[val[1] for val in top_words_positive]
data_positive.to_csv('sentiment_amazon_positive_unigram.csv', index = False)
'''
'''
filename = glob.glob("a_unigram_negative/*.pickle")
counter = Counter()
for i,file in enumerate(filename):
    print(i)
    with open(file, 'rb') as handle:
        b = pickle.load(handle)
        counter += b
top_words_negative = counter.most_common(50)
data_negative=pd.DataFrame()
data_negative['words']=[val[0] for val in top_words_negative]
data_negative['freq'] =[val[1] for val in top_words_negative]
data_negative.to_csv('sentiment_amazon_negative_unigram.csv', index = False)
'''
'''
filename = glob.glob("a_bigram_positive/*.pickle")
counter = Counter()
for i,file in enumerate(filename):
    print(i)
    with open(file, 'rb') as handle:
        b = pickle.load(handle)
        counter += b
top_words_positive = counter.most_common(50)
data_positive=pd.DataFrame()
data_positive['words']=[val[0] for val in top_words_positive]
data_positive['freq']=[val[1] for val in top_words_positive]
data_positive.to_csv('sentiment_amazon_positive_bigram.csv', index = False)
'''
'''
filename = glob.glob("a_bigram_negative/*.pickle")
counter = Counter()
for i,file in enumerate(filename):
    print(i)
    with open(file, 'rb') as handle:
        b = pickle.load(handle)
        counter += b
top_words_negative = counter.most_common(50)
data_negative=pd.DataFrame()
data_negative['words']=[val[0] for val in top_words_negative]
data_negative['freq'] =[val[1] for val in top_words_negative]
data_negative.to_csv('sentiment_amazon_negative_bigram.csv', index = False)
'''
'''
filename = glob.glob("a_trigram_positive/*.pickle")
counter = Counter()
for i,file in enumerate(filename):
    print(i)
    counter = Counter({k:v for k,v in counter.items() if v >1})
    with open(file, 'rb') as handle:
        b = pickle.load(handle)
        
        counter += b
top_words_positive = counter.most_common(50)
data_positive=pd.DataFrame()
data_positive['words']=[val[0] for val in top_words_positive]
data_positive['freq']=[val[1] for val in top_words_positive]
data_positive.to_csv('sentiment_amazon_positive_trigram.csv', index = False)
'''

filename = glob.glob("a_trigram_negative/*.pickle")
counter = Counter()
for i,file in enumerate(filename):
    print(i)
    counter = Counter({k:v for k,v in counter.items() if v >1})
    with open(file, 'rb') as handle:
        b = pickle.load(handle)
        counter += b
top_words_negative = counter.most_common(50)
data_negative=pd.DataFrame()
data_negative['words']=[val[0] for val in top_words_negative]
data_negative['freq']=[val[1] for val in top_words_negative]
data_negative.to_csv('sentiment_amazon_negative_trigram.csv', index = False)