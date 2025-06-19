import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

from .faiss_utils import faiss_search


def k_reciprocal_neigh(I, i, k):
    forward_k_neigh_index = I[i, : k]
    backward_k_neigh_index = I[forward_k_neigh_index, : k]
    fi = np.where(backward_k_neigh_index == i)[0]
    return set(forward_k_neigh_index[fi])


def jaccard_distance(fs, Nq=0, k1=20, k2=6, search_option=1):
    N = fs.size(0)
    I = faiss_search(fs, k = k1, search_option = search_option)

    nn_k1_half = []
    for i in range(N):
        nn_k1_half.append(k_reciprocal_neigh(I, i, int(np.around(k1 / 2))))

    V = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        k_reciprocal_index = k_reciprocal_neigh(I, i, k1)
        k_reciprocal_expansion_index = deepcopy(k_reciprocal_index)
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if len(candidate_k_reciprocal_index & k_reciprocal_index) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index.update(candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = torch.LongTensor(list(k_reciprocal_expansion_index))
        x = fs[i:i+1,:].contiguous()
        y = fs[k_reciprocal_expansion_index,:]
        dist = torch.cdist(x, y)**2
        V[i, k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).numpy()

    del nn_k1_half, x, y, k_reciprocal_expansion_index, dist

    if k2 != 1:
        V = np.mean(V[I[:, :k2], :], axis=1)

    del I

    if Nq:
        invIndex = []
        for i in range(N):
            invIndex.append(np.where(V[Nq:, i] != 0)[0])  # len(invIndex)=all_num
        
        jaccard_dist = np.zeros((Nq, N-Nq), dtype=np.float32)
        for i in range(Nq):
            temp_min = np.zeros((1, N-Nq), dtype=np.float32)
            indNonZero = np.where(V[i, :] != 0)[0]
            indImages = [invIndex[ind] for ind in indNonZero]
            for j in range(len(indNonZero)):
                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                    V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
                )

            jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
    else:
        invIndex = []
        for i in range(N):
            invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num
            
        jaccard_dist = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            temp_min = np.zeros((1, N), dtype=np.float32)
            indNonZero = np.where(V[i, :] != 0)[0]
            indImages = [invIndex[ind] for ind in indNonZero]
            for j in range(len(indNonZero)):
                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                    V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
                )

            jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    del invIndex, V

    pos_bool = jaccard_dist < 0
    jaccard_dist[pos_bool] = 0.0

    return jaccard_dist

"""
import timeit

N=20

setup = '''
import sys

from evaluation.rerank.jaccard_distance import jaccard_distance
from evaluation.rerank.jaccard_distance_ref import compute_jaccard_distance
import numpy as np
import torch

fs = torch.rand(500, 256)
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

import sys

from evaluation.rerank.jaccard_distance import jaccard_distance
from evaluation.rerank.jaccard_distance_ref import compute_jaccard_distance
import numpy as np
import torch

fs = torch.rand(500, 256)
Nq = 0
search_option = 1

d1 = jaccard_distance(fs, Nq, search_option=search_option)
d2 = compute_jaccard_distance(fs, search_option=search_option)

np.testing.assert_allclose(d1, d2)

"""