import timeit

N=10

setup = '''
import sys

from evaluation.rerank.jaccard_distance import jaccard_distance
from evaluation.rerank.jaccard_distance_ import compute_jaccard_distance
import numpy as np
import torch

fs = torch.rand(5000, 256)
Nq = 0
search_option = 1
'''
  
f1 = '''
d1 = jaccard_distance(fs, Nq, search_option=search_option)
'''

f2 = '''
d2 = compute_jaccard_distance(fs, search_option=search_option)
'''


time1 = timeit.timeit(
    f1,
    setup=setup,
    number=N
)
time2 = timeit.timeit(
    f2,
    setup=setup,
    number=N
)

print('Time1: {} s'.format(time1))
print('Time2: {} s'.format(time2))
