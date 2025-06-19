from .datasets import load_datasource, DataSet
from .samplers import load_sampler
from .transforms import load_transforms
from torch.utils.data import DataLoader
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


def build_dataloaders(args):
    (infos_trn, pids_trn, cids_trn), (infos_qry, pids_qry, cids_qry), (infos_gal, pids_gal, cids_gal) = load_datasource(
        args.dataset, args.dataset_trn, args.dataset_qry, args.dataset_gal, args.p_trn, args.split_mode_trn
    )
    
    tfs_trn = load_transforms(
        True, args.pixel_mean, args.pixel_std, size_train=args.img_size, size_test=args.img_size, **args.aa_tf, **args.crop_tf, **args.pad_tf, **args.flip_tf, **args.rea_tf
    )
    tfs_tst = load_transforms(
        False, args.pixel_mean, args.pixel_std, size_test=args.img_size,
    )
    
    pids_idxs_trn = dict([(pid, i) for i, pid in enumerate(sorted(pids_trn))])
    cids_idxs_trn = dict([(cid, i) for i, cid in enumerate(sorted(cids_trn))])
    ds_trn = DataSet(infos_trn, pids_idxs_trn, cids_idxs_trn, tfs_trn)
    
    pids_qry_gal = sorted(pids_qry | pids_gal)
    pids_idxs_qry_gal = dict([(pid, i) for i, pid in enumerate(pids_qry_gal)])
    
    cids_qry_gal = (cids_qry | cids_gal) - cids_trn # exclude query and gallery cids from training
    cids_idxs_qry_gal = deepcopy(cids_idxs_trn)
    if len(cids_qry_gal):
        cids_idxs_qry_gal.update(dict([(cid, i+len(cids_trn)) for i, cid in enumerate(sorted(cids_qry_gal))]))
    ds_tst = DataSet(infos_qry + infos_gal, pids_idxs_qry_gal, cids_idxs_qry_gal, tfs_tst)
    
    sp_trn = load_sampler(infos_trn, args.bs_trn, True, args.sp_forever)
    sp_tst = load_sampler(infos_qry + infos_gal, args.bs_tst, False)
    dl_trn = DataLoader(ds_trn, batch_size=sp_trn._batch_size, sampler=sp_trn, num_workers=args.num_workers, pin_memory=True)
    dl_tst = DataLoader(ds_tst, batch_size=sp_tst._batch_size, sampler=sp_tst, num_workers=args.num_workers, pin_memory=True)
    
    dl_tst.num_qry = len(infos_qry)
    dl_trn.num_cls = len(pids_trn)
    dl_trn.num_cid = len(cids_trn)
    
    return dl_trn, dl_tst
    
    