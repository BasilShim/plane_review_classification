import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

pos = pd.read_csv('positive.csv', sep=";", header=None) # positive tweets
neg = pd.read_csv('negative.csv', sep=";", header=None) # negative tweets
df = pos.append(neg, sort=True) # merge into one dataframe
print(df.shape) # size of df
df = df.drop(df.columns[[0, 1, 2, 5, 6, 7, 8, 9, 10, 11]], axis=1) # remove everything but text and sentiment 
df.columns = ['tweet text', 'tweet sentiment'] # add column names
print(df.shape) # shape of cleaned df
#df.to_csv(r'df.csv') # save to csv (incase)

#df = df.drop(df.columns[[1]], axis=1) # create df of just text
df['tweet text'] = df['tweet text'].str.replace("[^А-Яа-я'\' '[-]+", '') # remove evrything but Russian letters, spaces and '-' in between words

df['tweet text'] = df['tweet text'].str.lower()

import multiprocessing
cores = multiprocessing.cpu_count()

from gensim.models import Word2Vec

sent = [row.split() for row in df['tweet text']]

model = Word2Vec(sent, min_count=5,size=100,workers=cores-2, window=3, sg=1)
model.save("word2vec.model")