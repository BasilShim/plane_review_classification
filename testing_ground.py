import numpy as np
import pandas as pd
from gensim.models import Word2Vec

model = Word2Vec.load("word2vec.model")
#print(model.most_similar(positive=['ел','говорить'], negative=['есть'], topn=5)) # должно было выдать говорил. В вкабуляре он есть
print(model.most_similar('говорил'))
