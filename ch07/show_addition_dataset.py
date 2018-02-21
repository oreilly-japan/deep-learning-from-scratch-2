# coding: utf-8
import sys, os
sys.path.append(os.pardir)
from dataset import sequence


(x_train, t_train), (x_val, t_val) = \
    sequence.load_data("addition.txt")
char_to_id, id_to_char = sequence.get_vocab()

print(x_train.shape, t_train.shape)
print(x_val.shape, t_val.shape)
# (45000, 7) (45000, 5)
# (5000, 7) (5000, 5)

print(x_train[0])
print(t_train[0])
# [ 3  0  2  0  0 11  5]
# [ 6  0 11  7  5]
