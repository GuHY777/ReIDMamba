import timeit
import numpy as np

N=10

setup = '''
import sys
import os.path as osp
import numpy as np
sys.path.insert(0, osp.dirname(osp.abspath(__file__)) + '/../../..')

import faiss
import torch
from rerank.jaccard_distance import compute_jaccard_distance, jaccard_distance, jaccard_distance_np
from scipy.spatial.distance import cdist

k1 = 20
k2 = 6
features = torch.rand(100, 128)
'''
  
f1 = '''
j0=compute_jaccard_distance(features, k1, k2)
'''

f2 = '''
ori_D = cdist(features.numpy(), features.numpy(), 'euclidean')
I = np.argsort(ori_D, axis=1)
j1=jaccard_distance_np(ori_D, I, 0, k1, k2)
'''

f3 = '''
ori_D = torch.cdist(features, features, p=2)
I = torch.argsort(ori_D, axis=1)
j2=jaccard_distance(ori_D, I, 0, k1, k2)
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
time3 = timeit.timeit(
    f3,
    setup=setup,
    number=N
)

print('Time1: {} s'.format(time1))
print('Time2: {} s'.format(time2))
print('Time3: {} s'.format(time3))

import sys
import os.path as osp
import numpy as np
sys.path.insert(0, osp.dirname(osp.abspath(__file__)) + '/../../..')

import faiss
import torch
from evaluation.rerank.jaccard_distance_ref import compute_jaccard_distance, jaccard_distance, jaccard_distance_np
from scipy.spatial.distance import cdist

k1 = 20
k2 = 6
features = torch.rand(100, 128)

j0=compute_jaccard_distance(features, k1, k2)

ori_D = cdist(features.numpy(), features.numpy(), 'euclidean')
I = np.argsort(ori_D, axis=1)
j1=jaccard_distance_np(ori_D, I, 0, k1, k2)

ori_D = torch.cdist(features, features, p=2)
I = torch.argsort(ori_D, axis=1)
j2=jaccard_distance(ori_D, I, 0, k1, k2)


print(np.allclose(j0, j1))
print(np.allclose(j1, j2.numpy()))