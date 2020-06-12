import numpy as np
import multiprocessing as mp
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#% matplotlib inline

def sentiment_analyzer_scores(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    return score

def compound_score(text):
    comp=sentiment_analyzer_scores(text)
    return comp['compound'] # returns the compound score from the dictionary

def sentiment_category(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

                
def worker(filename):
    wfile = filename.split('.tsv')[0].split('_')[5] 
    if os.path.exists('sentiment_gr_category/done/'+ wfile +'.tsv'):
        return
    with open(filename,'r') as f:
        next(f)
        pending = []
        with open('sentiment_gr_category/sentiment_gr_category_'+ wfile +'.tsv', 'w') as g: 
            pending.append('overall' + '\t'  + 'asin'+'\t' + 'cleaned_text' + '\t'+ 'sentiment_score'+'\t'+ 'review_category'+'\n')
            #g.write('overall' + '\t' + 'reviewTime' + '\t' + 'asin'+'\t'+'reviewText' + '\t' + 'cleaned_text' + '\n')
            lines = f.readlines()
            for i,line in enumerate(lines):
                text = line.strip().split('\t')
                if(len(text) > 5):
                    overall = text[1]
                    asin    = text[3]
                    cleaned_text = text[5]
                    sentiment_score = compound_score(cleaned_text)
                    #print(sentiment_score)
                    review_category = sentiment_category(sentiment_score)
                    pending.append(overall + '\t' + asin + '\t' + cleaned_text + '\t' + str(sentiment_score) + '\t' + review_category + '\n' )
                    #g.write(overall + '\t' + reviewTime + '\t' + asin +'\t' + '' + '\t' + '' + '\n')
            g.write("".join(pending))
    with open('sentiment_gr_category/done/'+ wfile +'.tsv', 'w') as h : # Create a folder to mark a file is cleaned 
        h.write('done')

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
jobs = ['sentiment_gr_clean/sentiment_gr_cleaned_{}.tsv'.format(k) for k in range(starting, stopping)]
results = pool.map(worker, jobs)