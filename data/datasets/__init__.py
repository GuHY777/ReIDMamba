from data.datasets.datasource import DataSource, _name_base
import logging
from tabulate import tabulate
from PIL import Image
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class DataSet:
    def __init__(self, infos, pids_idxs, cids_idxs, tfs=None):
        self.infos = infos
        self.pids_idxs = pids_idxs
        self.cids_idxs = cids_idxs
        self.tfs = tfs
        
    def __len__(self):
        return len(self.infos)
    
    def __getitem__(self, idx):
        img_path, pid, cid = self.infos[idx]
        img = Image.open(img_path).convert('RGB')
        if self.tfs is not None: img = self.tfs(img)
        pid = self.pids_idxs[pid]
        cid = self.cids_idxs[cid]
        return img, pid, cid


def _load_datasource(dataset_names, ds):
    for dn in dataset_names:
        d = dn.split('_')[0]
        if d not in _name_base:
            raise ValueError(f'Invalid dataset name: {dn}!')
        
        if d not in ds:
            logger.info(f'Load {d} datasource.')
            ds[d] = DataSource(d)
            
    return ds

def load_infos_pids_cids(dataset_names, ds):
    infos = []
    pids = set()
    cids = set()
    for dn in dataset_names:
        d, t = dn.split('_')
        if t == 'Trn':
            infos.extend(ds[d].infos_trn)
            pids.update(ds[d].pids_trn)
            cids.update(ds[d].cids_trn)
        elif t == 'Qry':
            infos.extend(ds[d].infos_qry)
            pids.update(ds[d].pids_qry)
            cids.update(ds[d].cids_qry)
        elif t == 'Gal':
            infos.extend(ds[d].infos_gal)
            pids.update(ds[d].pids_gal)
            cids.update(ds[d].cids_gal)
        elif t == 'All':
            infos.extend(ds[d].infos_trn)
            pids.update(ds[d].pids_trn)
            cids.update(ds[d].cids_trn)
            
            infos.extend(ds[d].infos_qry)
            pids.update(ds[d].pids_qry)
            cids.update(ds[d].cids_qry)
            
            infos.extend(ds[d].infos_gal)
            pids.update(ds[d].pids_gal)
            cids.update(ds[d].cids_gal)
        else:
            raise ValueError(f'Invalid dataset name: {dn}!')
    
    return infos, pids, cids

def load_datasource(dataset_names, names_trn, names_qry, names_gal, p_trn=1.0, split_mode='by_person'):
    assert isinstance(dataset_names, list), f'The arg of dataset_names must be a list, but got {type(dataset_names)}!'
    if '' not in dataset_names:
        names_trn = tuple(dn + '_Trn' for dn in dataset_names)
        names_qry = tuple(dn + '_Qry' for dn in dataset_names)
        names_gal = tuple(dn + '_Gal' for dn in dataset_names)
    else:
        assert '' not in names_trn, f'The name of training set cannot be empty!'
        assert '' not in names_qry, f'The name of query set cannot be empty!'
        assert '' not in names_gal, f'The name of gallery set cannot be empty!'
    
    ds = {}
    ds = _load_datasource(names_trn, ds)
    ds = _load_datasource(names_qry, ds)
    ds = _load_datasource(names_gal, ds)
    
    _infos_trn, _pids_trn, _cids_trn = load_infos_pids_cids(names_trn, ds)
    if p_trn < 1.0:
        if split_mode == 'by_person':
            logger.info(f'Split training set by person with p_trn={p_trn}.')
            res_pids = np.random.choice(list(_pids_trn), max(1, int(len(_pids_trn)*p_trn)), replace=False)
            infos_trn = [info for info in _infos_trn if info[1] in res_pids]
            pids_trn = set(info[1] for info in infos_trn)
            cids_trn = set(info[2] for info in infos_trn)
        elif split_mode == 'by_camera':
            logger.info(f'Split training set by camera with p_trn={p_trn}.')
            res_cids = np.random.choice(list(_cids_trn), max(1, int(len(_cids_trn)*p_trn)), replace=False)
            infos_trn = [info for info in _infos_trn if info[2] in res_cids]
            pids_trn = set(info[1] for info in infos_trn)
            cids_trn = set(info[2] for info in infos_trn)
        elif split_mode == 'in_person':
            logger.info(f'Split training set in person with p_trn={p_trn}.')
            _pid_idxs = defaultdict(list)
            for idx, info in enumerate(_infos_trn):
                _pid_idxs[info[1]].append(idx)
            
            res_idxs = []
            for pid in _pid_idxs:
                res_idxs.extend(np.random.choice(_pid_idxs[pid], max(1, int(len(_pid_idxs[pid])*p_trn)), replace=False))
            
            infos_trn = [info for idx, info in enumerate(_infos_trn) if idx in res_idxs]
            pids_trn = set(info[1] for info in infos_trn)
            cids_trn = set(info[2] for info in infos_trn)
        else:
            raise ValueError(f'Invalid split_mode: {split_mode}!')
    else:
        infos_trn = _infos_trn
        pids_trn = _pids_trn
        cids_trn = _cids_trn

    infos_qry, pids_qry, cids_qry = load_infos_pids_cids(names_qry, ds)
    infos_gal, pids_gal, cids_gal = load_infos_pids_cids(names_gal, ds)
    
    data = [
        [f"TRAIN:{'+'.join(names_trn)}", len(infos_trn), len(pids_trn), len(cids_trn)],
        [f"QUERY:{'+'.join(names_qry)}", len(infos_qry), len(pids_qry), len(cids_qry)],
        [f"GALLERY:{'+'.join(names_gal)}", len(infos_gal), len(pids_gal), len(cids_gal)]]
    table = tabulate(data, headers=['Split', '#Images', '#PIDs', '#cids'], tablefmt='grid')
    logger.info(f'# --- Dataset --- #\n{table}')
    
    return (infos_trn, pids_trn, cids_trn), (infos_qry, pids_qry, cids_qry), (infos_gal, pids_gal, cids_gal)
        
    
    