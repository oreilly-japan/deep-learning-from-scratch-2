# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import matplotlib.pyplot as plt
import numpy as np
from common.util import most_similar, analogy
import pickle

pkl_file = 'word2vec_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs'].astype('f')
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

W = word_vecs
U, S, V = np.linalg.svd(W)


plt_words = ['man', 'woman', 'king', 'queen', 'brother', 'sister']
for word in plt_words:
    word_id = word_to_id[word]
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
    plt.scatter(U[word_id, 0], U[word_id, 1], alpha=0.5, color='green')


plt.show()
#plt.savefig('count_base.eps', format='eps')