from torch.utils.data.sampler import Sampler
from .balancedsampler import BalancedIdentitySampler
import numpy as np
import random
import torch

import logging
logger = logging.getLogger(__name__)


def load_sampler(infos, batch_size, is_train=True, forever=True):
    if len(batch_size) == 2:
        sp = BalancedIdentitySampler(infos, batch_size, forever)
    else:
        if is_train:
            sp = TrainingSampler(infos, batch_size[0], forever)
        else:
            sp = InferenceSampler(infos, batch_size[0])
    return sp


class TrainingSampler(Sampler):
    def __init__(self, infos, batch_size, forever=True, shuffle=True):
        super().__init__()
        self._batch_size = batch_size
        self._size = len(infos)
        self.forever = forever
        self.shuffle = shuffle
        self.nums_epoch = len(infos) // self._batch_size if len(infos) % self._batch_size == 0 else len(infos) // self._batch_size + 1
        
    def __iter__(self):
        if self.forever:
            while True:
                yield from self.get_idxs()
        else:
            idxs = self.get_idxs()
            while len(idxs) < self.nums_epoch * self._batch_size:
                idxs = np.concatenate([idxs, self.get_idxs()])
            yield from idxs[:self.nums_epoch * self._batch_size]
                
    def get_idxs(self):
        if self.shuffle:
            indices = np.random.permutation(self._size)
        else:
            indices = np.arange(self._size)
        return indices
            
    def set(self, s):
        if s > -1:
            logger.info(f"Setting {self.__class__.__name__} seed to {s}")
            torch.manual_seed(s)
            torch.cuda.manual_seed(s)
            torch.cuda.manual_seed_all(s)
            np.random.seed(s)
            random.seed(s)
            
            
class InferenceSampler(Sampler):
    def __init__(self, infos, batch_size):
        super().__init__()
        self._batch_size = batch_size
        self._size = len(infos)
        
    def __iter__(self):
        indices = np.arange(self._size)
        yield from indices