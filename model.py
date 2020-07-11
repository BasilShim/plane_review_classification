import numpy as np
import pandas as pd
import time
import multiprocessing
import nltk.data
import nltk
import collections
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

pd.set_option('display.max_rows', None) # pandas print options are set to show the full dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

def split_into_sentences(df):  # hepler function for splitting sentences
    count = len(df.index)
    sentences = []
    i = 1
    print("\nSplitting into sentences. Please, stand by.\n")
    for row in df.itertuples():
        for sentence in nltk.sent_tokenize(row[1]):
            sentences.append((sentence, row[2]))
        print("Processed %s out of %s rows" % (i, count), end="\r", flush=True)
        i += 1
 
    return pd.DataFrame(sentences, columns=['text', 'sentiment'])

def remove_stop_words(df):  # hepler function for removing stop words
    stop_words = set(nltk.corpus.stopwords.words('russian'))
    stop_words.remove("хорошо")
    sent_no_sw = []
    count = len(df.index)
    i = 1
    print("\nRemoving stop words. Please, stand by.\n")
    for row in df.itertuples():
        tokens = word_tokenize(row[1])
        result = [i for i in tokens if not i in stop_words]
        sep = " "
        result = sep.join(result)
        sent_no_sw.append((result, row[2]))
        print("Processed %s out of %s rows" % (i, count), end="\r", flush=True)
        i += 1
        
    return pd.DataFrame(sent_no_sw, columns=['text', 'sentiment'])

def get_statistics(sentences): # print some statistics 
    unique_words = []
    unique_words = set(word for sent in sentences for word in sent)
    print("\nNumber of unique words:\n")
    print(len(unique_words)) # print number of unique words

    word_freq = collections.defaultdict(int)
    for sent in sentences:
        for i in sent:
            word_freq[i] += 1
    len(word_freq)
    print("\nTop 10 most frequent words:\n")
    print(sorted(word_freq, key=word_freq.get, reverse=True)[:10]) # top 10 most frequent words
    return None

def main():
    start_time = time.time()
    pos = pd.read_csv('positive.csv', sep=";", header=None) # positive tweets
    neg = pd.read_csv('negative.csv', sep=";", header=None) # negative tweets
    df = pos.append(neg, sort=True) # merge into one dataframe
    print(df.shape) # size of df
    df = df.drop(df.columns[[0, 1, 2, 5, 6, 7, 8, 9, 10, 11]], axis=1) # remove everything but text and sentiment 
    df.columns = ['text', 'sentiment'] # add column names
    print(df.shape) # shape of cleaned df
    #df.to_csv(r'df.csv') # save to csv (just in case)

    df = split_into_sentences(df)
    df.to_csv(r'sentences.csv', index=False) # save to csv to never ever do this again

    df['text'] = df['text'].str.replace("[-]", ' ')
    df['text'] = df['text'].str.replace("ё", 'е')
    df['text'] = df['text'].str.replace("Ё", 'Е')
    df['text'] = df['text'].str.replace("[^А-Яа-я'\' ']+", '') # remove evrything but Russian letters and spaces

    df['text'] = df['text'].str.lower() # convert everything to lowercase letters

    sentences = [row.split() for row in df['text']]

    df = remove_stop_words(df)
    df.to_csv(r'sent_no_sw.csv', index=False) # save to csv to never ever do this again

    sent = [row.split() for row in df['text']]

    phrases = Phrases(sent, min_count=20) # train a bigram model to detect phrases that occur more than 20 times
    bigram = Phraser(phrases)
    sentences = bigram[sent] # transform the corpus based on the bigrams detected

    get_statistics(sentences)

    cores = multiprocessing.cpu_count() # count number of cores

    print("\nBuilding the model. Please, stand by.\n")
    model = Word2Vec(sentences, min_count=5, size=100, workers=cores-2, window=3, sg=1) # the model itself
    print("\nDone building. Now saving the model. Please, stand by.\n")
    model.save("word2vec.model")
    print("The whole process took %s seconds" % (time.time() - start_time))
    return None

main()
