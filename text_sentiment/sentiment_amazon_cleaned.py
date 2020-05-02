import sys
#cell 3
#import pandas as pd
import numpy as np
from textblob import TextBlob, Word
import string
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer
import re
from nltk.stem import WordNetLemmatizer
from autocorrect import Speller
import multiprocessing as mp
import os

#cell 4
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
STOPWORDS = stopwords.words("english") #stopwords are the most common unnecessary words. eg is, he, that, etc.

#cell 5
def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii') # A function to remove emojis from the reviews

#cell 6
def lemmatize_with_postag(sentence, wnl):
    sent = TextBlob(sentence)
    tag_dict = {"J": 'a', 
                "N": 'n', 
                "V": 'v', 
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    #lemmatized_list = [wnl.lemmatize(wd, tag) for wd, tag in words_and_tags]
    return " ".join(lemmatized_list)
#cell 7
spell = Speller(lang='en')
contractions_dict = {     
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I had",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "iit will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that had",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there had",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they had",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}
def expand_contractions(text, contractions_dict):
    contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contractions_dict.get(match) \
            if contractions_dict.get(match) \
            else contractions_dict.get(match.lower())
        expanded_contraction = expanded_contraction
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

#cell 8
# Remove 'not' for sentiment analysis
STOPWORDS.remove('not')
STOPWORDS.remove('no')
STOPWORDS.remove('nor')
STOPWORDS.remove('but')
STOPWORDS.remove('very')
STOPWORDS.append('will')
len(STOPWORDS)
punctuation = ['.',',','"','$','%','&','`']
#cell 9
def clean_text(text, wnl):
    if not isinstance(text, str):
        return ''
    text_cleaned=re.sub(' +', ' ', text) # remove extra white spaces
    text_cleaned=text_cleaned.lower() # converting to lowercase
    text_cleaned = ''.join(c for c in text_cleaned if not c.isdigit())# remove numbers
    text_cleaned = expand_contractions(text_cleaned, contractions_dict) # contraction 
    text_cleaned="".join([x for x in text_cleaned if x not in punctuation]) # remove some punctuations
    
    text_cleaned = nltk.word_tokenize(text_cleaned)
    text_cleaned = [x for x in text_cleaned if len(x) < 20]
    #text_cleaned = ' '.join(spell(w) for w in (text_cleaned))
    text_cleaned = [spell(w) for w in (text_cleaned)]   # correct spelling
    #text_cleaned=text_cleaned.split(" ")
    text_cleaned=" ".join([token for token in text_cleaned if token not in STOPWORDS]) # Taking only those words which are not stopwords
    
    #Converting to lemma
    text_cleaned = lemmatize_with_postag(str(text_cleaned), wnl)

    return text_cleaned

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

#cell 12
def worker(filename):
    wfile = filename.split('.tsv')[0].split('_')[3] 
    if os.path.exists('sentiment_amazon_clean/done/'+ wfile +'.tsv'):
        return
    wnl = WordNetLemmatizer()
    with open(filename,'r') as f:
        next(f)
        pending = []
        with open('sentiment_amazon_clean/sentiment_amazon_cleaned_'+ wfile +'.tsv', 'w') as g: 
            pending.append('overall' + '\t' + 'reviewTime' + '\t' + 'asin'+'\t'+'reviewText' + '\t' + 'cleaned_text' + '\n')
            #g.write('overall' + '\t' + 'reviewTime' + '\t' + 'asin'+'\t'+'reviewText' + '\t' + 'cleaned_text' + '\n')
            lines = f.readlines()
            for i, line in enumerate(lines):
                #if i == 3:
                #print(i,line)    #break
                line = line.strip().split('\t')
                overall = line[0]
                reviewTime = line[1]
                asin = line[2]
                if len(line) > 3:
                    reviewText = line[3]
                    reviewText = remove_tags(reviewText)
                    cleaned_text = clean_text(reviewText, wnl)
                    pending.append(overall + '\t' + reviewTime + '\t' + asin +'\t' + reviewText + '\t' + cleaned_text + '\n')
                    #g.write(overall + '\t' + reviewTime + '\t' + asin +'\t' + reviewText + '\t' + cleaned_text + '\n')
                else:
                    pending.append(overall + '\t' + reviewTime + '\t' + asin +'\t' + '' + '\t' + '' + '\n')
                    #g.write(overall + '\t' + reviewTime + '\t' + asin +'\t' + '' + '\t' + '' + '\n')
            g.write("".join(pending))
    with open('sentiment_amazon_clean/done/'+ wfile +'.tsv', 'w') as h : # Create a folder to mark a file is cleaned 
        h.write('done')

#cell 13
#Performing multiprocessing to speed up
"""
import time
t0 = time.time()
num_cpu = 20  # number of CPU will be used
pool = mp.Pool(num_cpu)
jobs = ['amazon_official/amazon_official_{}.tsv'.format(k) for k in range(1,501)] # there are 500 files total
#jobs = ['amazon_official/amazon_official_001.tsv','amazon_official/amazon_official_002.tsv']
results = pool.map(worker,jobs)
print(time.time()-t0)
"""
#cell 14
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
jobs = ['amazon_official/amazon_official_{}.tsv'.format(k) for k in range(starting, stopping)]
results = pool.map(worker, jobs)

#cell 15


