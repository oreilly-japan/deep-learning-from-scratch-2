# coding: utf-8
import sys
import os
sys.path.append(os.pardir)
from dataset import ptb


corpus, word_to_id, id_to_word = ptb.load_data("train")

print("corpus size:", len(corpus))
print("corpus[:30]:", corpus[:30])

print("\nid_to_word[0]:", id_to_word[0])
print("id_to_word[1]:", id_to_word[1])
print("id_to_word[2]:", id_to_word[2])

print("\nword_to_id['car']:", word_to_id["car"])
print("word_to_id['happy']:", word_to_id["happy"])
print("word_to_id['lexus']:", word_to_id["lexus"])
