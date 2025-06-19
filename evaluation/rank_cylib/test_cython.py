import sys
import timeit
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F

sys.path.insert(0, osp.dirname(osp.abspath(__file__)) + '/../../..')

from rank_cylib import evaluate_rank

"""
Test the speed of cython-based evaluation code. The speed improvements
can be much bigger when using the real reid data, which contains a larger
amount of query and gallery images.
Note: you might encounter the following error:
  'AssertionError: Error: all query identities do not appear in gallery'.
This is normal because the inputs are random numbers. Just try again.
"""

print('*** Compare running time ***')

setup = '''
import sys
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F 
sys.path.insert(0, osp.dirname(osp.abspath(__file__)) + '/../../..')
from rank_cylib import evaluate_rank

num_q = 30
num_g = 300
dim = 512
max_rank = 5
q_feats = torch.rand(num_q, dim) * 20
q_feats = F.normalize(q_feats, dim=1, p=2)
g_feats = torch.rand(num_g, dim) * 20
g_feats = F.normalize(g_feats, dim=1, p=2)
distmat = torch.cdist(q_feats, g_feats, p=2)
indices = torch.argsort(distmat, dim=1).numpy()
q_pids = np.random.randint(0, num_q, size=num_q)
g_pids = np.random.randint(0, num_g, size=num_g)
q_camids = np.random.randint(0, 5, size=num_q)
g_camids = np.random.randint(0, 5, size=num_g)
'''

print('=> Using CMC metric')
pytime = timeit.timeit(
    'evaluate_rank(indices, q_pids, g_pids, q_camids, g_camids, max_rank, use_cython=False)',
    setup=setup,
    number=20
)
cytime = timeit.timeit(
    'evaluate_rank(indices, q_pids, g_pids, q_camids, g_camids, max_rank, use_cython=True)',
    setup=setup,
    number=20
)

print('Python time: {} s'.format(pytime))
print('Cython time: {} s'.format(cytime))
print('CMC Cython is {} times faster than python\n'.format(pytime / cytime))

print("=> Check precision")
num_q = 30
num_g = 300
dim = 512
max_rank = 5
q_feats = torch.rand(num_q, dim) * 20
q_feats = F.normalize(q_feats, dim=1, p=2)
g_feats = torch.rand(num_g, dim) * 20
g_feats = F.normalize(g_feats, dim=1, p=2)
distmat = torch.cdist(q_feats, g_feats, p=2)
indices = torch.argsort(distmat, dim=1).numpy()
q_pids = np.random.randint(0, num_q, size=num_q)
g_pids = np.random.randint(0, num_g, size=num_g)
q_camids = np.random.randint(0, 5, size=num_q)
g_camids = np.random.randint(0, 5, size=num_g)

cmc_py, mAP_py, mINP_py = evaluate_rank(indices, q_pids, g_pids, q_camids, g_camids, max_rank, use_cython=False)

cmc_cy, mAP_cy, mINP_cy = evaluate_rank(indices, q_pids, g_pids, q_camids, g_camids, max_rank, use_cython=True)

np.testing.assert_allclose(cmc_py, cmc_cy, rtol=1e-3, atol=1e-6)
np.testing.assert_allclose(mAP_py, mAP_cy, rtol=1e-3, atol=1e-6)
np.testing.assert_allclose(mINP_py, mINP_cy, rtol=1e-3, atol=1e-6)

print("Python:")
print(f"CMC: {cmc_py}\nmAP: {mAP_py}\nmINP: {mINP_py}")
print("\nCython:")
print(f"CMC: {cmc_cy}\nmAP: {mAP_cy}\nmINP: {mINP_cy}")

