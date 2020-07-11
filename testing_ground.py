import numpy as np
import pandas as pd
from gensim.models import Word2Vec

model = Word2Vec.load("word2vec.model")
print(model.most_similar(positive=['хорошо','плохой'], negative=['хороший'], topn=5))
print(model.most_similar('сложный'))
