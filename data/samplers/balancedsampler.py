from copy import deepcopy
import itertools
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler
import torch
import random

import logging
logger = logging.getLogger(__name__)

class BalancedIdentitySampler(Sampler):
    """
    Balanced Identity Sampler.
    """
    def __init__(self, infos, batch_size, forever=True) -> None:
        super().__init__()
        self.num_ids, self.num_ins = batch_size
        self._batch_size = self.num_ids * self.num_ins
        self.infos = infos
        self.forever = forever
        self.nums_epoch = len(self.infos) // self._batch_size if len(self.infos) % self._batch_size == 0 else len(self.infos) // self._batch_size + 1
        
        self.pid_idx = defaultdict(list)
        for idx, info in enumerate(self.infos):
            pid = info[1]
            self.pid_idx[pid].append(idx)
        
        self.pids = sorted(list(self.pid_idx.keys()))

    def __iter__(self):
        if self.forever:
            while True:
                avl_pids = deepcopy(self.pids)
                batch_idxs_dict = {}

                batch_indices = []
                while len(avl_pids) >= self.num_ids:
                    selected_pids = np.random.choice(avl_pids, self.num_ids, replace=False).tolist()
                    for pid in selected_pids:
                        # Register pid in batch_idxs_dict if not
                        if pid not in batch_idxs_dict:
                            idxs = deepcopy(self.pid_idx[pid])
                            if len(idxs) < self.num_ins:
                                idxs = np.random.choice(idxs, size=self.num_ins, replace=True).tolist()
                            np.random.shuffle(idxs)
                            batch_idxs_dict[pid] = idxs

                        avl_idxs = batch_idxs_dict[pid]
                        for _ in range(self.num_ins):
                            batch_indices.append(avl_idxs.pop(0))

                        if len(avl_idxs) < self.num_ins: avl_pids.remove(pid)

                    # assert len(batch_indices) == self._batch_size
                    yield from batch_indices
                    batch_indices = []
        else:
            self.i = 0
            while self.i < self.nums_epoch:
                avl_pids = deepcopy(self.pids)
                batch_idxs_dict = {}

                batch_indices = []
                while len(avl_pids) >= self.num_ids:
                    if self.i == self.nums_epoch:
                        break
                    
                    selected_pids = np.random.choice(avl_pids, self.num_ids, replace=False).tolist()
                    for pid in selected_pids:
                        # Register pid in batch_idxs_dict if not
                        if pid not in batch_idxs_dict:
                            idxs = deepcopy(self.pid_idx[pid])
                            if len(idxs) < self.num_ins:
                                idxs = np.random.choice(idxs, size=self.num_ins, replace=True).tolist()
                            np.random.shuffle(idxs)
                            batch_idxs_dict[pid] = idxs

                        avl_idxs = batch_idxs_dict[pid]
                        for _ in range(self.num_ins):
                            batch_indices.append(avl_idxs.pop(0))

                        if len(avl_idxs) < self.num_ins: avl_pids.remove(pid)

                    # assert len(batch_indices) == self._batch_size
                    yield from batch_indices
                    batch_indices = []
                    
                    self.i += 1
                    
                    
    
    def set(self, s):
        if s > -1:
            logger.info(f"Setting {self.__class__.__name__} seed to {s}")
            torch.manual_seed(s)
            torch.cuda.manual_seed(s)
            torch.cuda.manual_seed_all(s)
            np.random.seed(s)
            random.seed(s)
    
    