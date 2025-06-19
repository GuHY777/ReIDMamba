import logging
import os
import os.path as osp
from typing import Dict

import torch
from ray.tune import Trainable
from torch import optim

from data import build_dataloaders
from evaluation import Evaluator
from optims import build_optimizer
from losses import Loss
from model import build_model
from optims.lr_schedulers import LinearWarmupLrScheduler
from utils import AverageMeter, setup
from utils.seed import setup_seed
import numpy as np

import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

logger = logging.getLogger(__name__)


class Trainer(Trainable):
    def setup(self, cfg, args):
        self.args = args
        
        # self.args.optim_kwargs['weight_decay'] = cfg['weight_decay']
        # self.args.model_kwargs['drop_path_rate'] = cfg['drop_path_rate']
        self.args.loss_kwargs[0]['margin'] = cfg['margin']
        self.args.loss_kwargs[1]['label_smoothing'] = cfg['label_smoothing']
        
        self.args.model_kwargs['stride_size'] = cfg['cfgs'][2]
        # self.args.model_kwargs['shuffle_mode'] = cfg['shuffle_mode']
        self.args.model_kwargs['drop_path_rate'] = cfg['drop_path_rate']
        # self.args.model_kwargs['num_cls_tokens'] = cfg['cfgs'][1]
        # self.args.model_kwargs['cls_reduce']     = cfg['cfgs'][2]
        self.args.img_size[0] = cfg['cfgs'][1]
        args.seed = cfg['seed']
        self.args.dataset[0] = cfg['cfgs'][0]
        
        # self.args.loss_weights = [1.0, 1.0, cfg['cfgs'][0], cfg['cfgs'][0]]
        self.args.loss_kwargs[2]['tau'] = cfg['tau']
        self.args.loss_kwargs[3]['tau'] = cfg['tau']
        self.thereshold = 1.2
        
        if self.args.dataset[0] == 'Market1501':
            self.args.lr_scheduler_kwargs['warmup_iters'] = 1000
        elif self.args.dataset[0] == 'DukeMTMC':
            self.args.lr_scheduler_kwargs['warmup_iters'] = 1200
        elif 'CUHK03-' in self.args.dataset[0]:
            self.args.lr_scheduler_kwargs['warmup_iters'] = 1000
        elif self.args.dataset[0] == 'MSMT17':
            self.args.lr_scheduler_kwargs['warmup_iters'] = 2500
        elif self.args.dataset[0] == 'OccludedDuke':
            self.args.lr_scheduler_kwargs['warmup_iters'] = 1200
        else:
            raise ValueError('dataset not supported')


        setup_seed(args.seed, True, True)
        if args.sp_seeds:
            self.seeds = np.random.randint(0, 1000000, args.epochs).tolist()
        else:
            self.seeds = [-1] * args.epochs
        
        os.environ['DATA_ROOT'] = args.dataroot
        self.dl_trn, self.dl_tst = build_dataloaders(self.args)
        self.model = build_model(self.args, self.dl_trn.num_cls, self.dl_trn.num_cid)
        self.model.to('cuda')

        self.optim, self.lrs = build_optimizer(self.args, self.model)

        self.loss = Loss(self.args)
        self.loss.to('cuda')
        
        self.evaluator = Evaluator(self.args, self.model, self.dl_tst)
        
        self.epoch = 0
        self.nums_epoch = self.dl_trn.sampler.nums_epoch
        self.iter_count = 0
        self.best_mAP = 0.0
        assert len(self.args.eval_freq) == 1
        self.eval_freq = int(self.args.eval_freq[0])
        
        # if cfg['margin']:
        #     self.thereshold = cfg['margin']
        # else:
        #     self.thereshold = F.softplus(torch.Tensor([0.0])).item()
            
        self.amp = self.args.amp
        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler(2**(10.0)) # 2**(10.0)
        
    def step(self):
        self.model.train()
        for epoch in range(self.epoch, self.epoch + self.eval_freq):
            
            losses_avg = AverageMeter()
            self.dl_trn.sampler.set(self.seeds[epoch])
            
            for i, (imgs, pids, cids) in enumerate(self.dl_trn):
                imgs = imgs.to('cuda')
                pids = pids.to('cuda')
                if self.model.use_cid:
                    cids_kwargs = {'cids': cids.to('cuda')}
                else:
                    cids_kwargs = {}
                    
                with torch.autocast(device_type='cuda', enabled=self.args.amp):
                    outputs = self.model(imgs, **cids_kwargs)            
                    loss_val, losses = self.loss(outputs, pids)  
                losses_avg(losses)
                
                if torch.isnan(loss_val):
                    raise ValueError('loss is nan, please check the model')
                
                if self.args.amp:
                    self.scaler.scale(loss_val).backward()
                    self.scaler.unscale_(self.optim)
                    if self.args.grad_clip > 0:
                        norm = clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    else:
                        norm = 0.0

                    self.scaler.step(self.optim)
                    self.scaler.update()
                else:
                    loss_val.backward()
                    
                    if self.args.grad_clip > 0:
                        norm = clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    else:
                        norm = 0.0
                    
                    self.optim.step()
                self.optim.zero_grad()
                
                self.iter_count += 1
                if self.iter_count <= self.args.lr_scheduler_kwargs['warmup_iters']:
                    self.lrs.step()
                
                if (i + 1) == self.nums_epoch and self.args.sp_forever:
                    break   
            
            if self.iter_count > self.args.lr_scheduler_kwargs['warmup_iters']:
                self.lrs.step()
                if losses_avg.avgs['triplet_loss'] > self.thereshold:
                    raise ValueError('triplet loss is too large, please check the model')
            
        self.epoch += self.eval_freq
        metrics, metrics_flip = self.evaluator(self.epoch)
        if metrics_flip is None:
            metrics_flip = {'mAP': 0.0, 'Rank-1': 0.0}
        
        if metrics['mAP'] > self.best_mAP:
            self.best_mAP = metrics['mAP']
            
        if metrics_flip['mAP'] > self.best_mAP:
            self.best_mAP = metrics_flip['mAP']
        
        # if max(metrics['mAP'], metrics_flip['mAP']) < 80.0:
        #     return {'best_mAP': self.best_mAP, 'mAP': metrics['mAP'], 'Rank-1': metrics['Rank-1'], 'mAP_flip':metrics_flip['mAP'], 'Rank-1_flip':metrics_flip['Rank-1'], 'should_checkpoint': False, 'done': True}
            
        if self.epoch == self.args.epochs:
        # if self.epoch == 40:
            return {'best_mAP': self.best_mAP, 'mAP': metrics['mAP'], 'Rank-1': metrics['Rank-1'], 'mAP_flip':metrics_flip['mAP'], 'Rank-1_flip':metrics_flip['Rank-1'], 'should_checkpoint': True, 'done': True}
        else:
            return {'best_mAP': self.best_mAP, 'mAP': metrics['mAP'], 'Rank-1': metrics['Rank-1'], 'mAP_flip':metrics_flip['mAP'], 'Rank-1_flip':metrics_flip['Rank-1'], 'should_checkpoint': True, 'done': False}
            
    def save_checkpoint(self, checkpoint_dir):
        ckpt_path = osp.join(checkpoint_dir, "ckpt.pth")
        ckpt = {
            'epoch': self.epoch,
            'iter_count': self.iter_count,
            'best_mAP': self.best_mAP,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'lr_scheduler_state_dict': self.lrs.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(ckpt, ckpt_path)
        
    def load_checkpoint(self, checkpoint_dir):
        ckpt_path = osp.join(checkpoint_dir, "ckpt.pth")
        ckpt = torch.load(ckpt_path)
        self.epoch = ckpt['epoch']
        self.iter_count = ckpt['iter_count']
        self.best_mAP = ckpt['best_mAP']
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optim.load_state_dict(ckpt['optimizer_state_dict'])
        self.lrs.load_state_dict(ckpt['lr_scheduler_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler'])
        
        
        
        
        
        