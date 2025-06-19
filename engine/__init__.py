import torch
import logging
from collections import defaultdict
import os.path as osp
from utils import AverageMeter
import numpy as np
from torch.nn.utils import clip_grad_norm_



logger = logging.getLogger(__name__)


class Engine:
    def __init__(self, seeds, args, model, model_ema, optim, lrs, loss, evaluator, dl_trn, dl_tst, tb_writer, savedir) -> None:
        self.seeds = seeds
        self.args = args
        self.model = model
        self.model_ema = model_ema
        self.optim = optim
        self.lrs = lrs
        self.loss = loss
        self.evaluator = evaluator
        self.dl_trn = dl_trn
        self.dl_tst = dl_tst
        self.tb_writer = tb_writer
        self.savedir = savedir
        if self.args.amp:
            logger.info("\nUsing Automatic Mixed Precision (AMP)")
            self.scaler = torch.cuda.amp.GradScaler(2**(10.0)) # 2**(10.0)

        self.nums_epoch = self.dl_trn.sampler.nums_epoch
            
        if len(self.args.eval_freq) == 1:
            self.eval_epochs = [self.args.eval_freq[0] for _ in range(self.args.epochs // self.args.eval_freq[0])]
            self.eval_epochs =  np.cumsum(self.eval_epochs).tolist()
            if self.args.epochs % self.args.eval_freq[0]:
                self.eval_epochs.append(self.args.epochs)
        else:
            assert self.args.eval_freq[-1] == self.args.epochs
            self.eval_epochs = self.args.eval_freq
            
        self.epoch = 0
        self.iter_count = 0
        
        
    def train_one_epoch(self, show_nums=50):
        # freeze or unfreeze the backbone
        if self.args.freeze_bb and self.epoch == 0 and hasattr(self.model, 'freeze_backbone'):
            self.model.freeze_backbone()
            logger.info("Backbone is frozen")
        
        if self.args.freeze_bb and self.epoch == self.args.freeze_bb and hasattr(self.model, 'unfreeze_backbone'):
            self.model.unfreeze_backbone()
            logger.info("Backbone is unfrozen")
            
        if self.args.eval_bb and self.epoch == 0 and hasattr(self.model, 'eval_backbone'):
            assert self.args.eval_bb <= self.eval_epochs[0], "eval_bb should be less than or equal to the first eval_freq!"
            self.model.eval_backbone()
            logger.info("Backbone is in eval mode")
        
        if self.args.eval_bb and self.epoch == self.args.eval_bb and hasattr(self.model, 'train_backbone'):
            self.model.train_backbone()
            logger.info("Backbone is in train mode")
            
        losses_avg = AverageMeter()
        gradnorm_avg = AverageMeter()
        
        self.dl_trn.sampler.set(self.seeds[self.epoch])
        
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
                return False
            
            if self.args.amp:
                self.scaler.scale(loss_val).backward()
                self.scaler.unscale_(self.optim)
                if self.args.grad_clip > 0:
                    norm = clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    gradnorm_avg({'gradnorm': norm.item()})
                else:
                    norm = 0.0

                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                loss_val.backward()
                
                # t = []
                # for p in self.model.parameters():
                #     if p.requires_grad and p.grad is not None:
                #         t.append(p.grad.detach().view(-1))
                # gradnorm_avg({'gradnorm': torch.cat(t).norm().item()})
                
                if self.args.grad_clip > 0:
                    norm = clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                else:
                    norm = 0.0
                gradnorm_avg({'gradnorm': norm.item()})
                
                self.optim.step()
            self.optim.zero_grad()
            
            if self.model_ema is not None:
                self.model_ema.update(self.model)
            
            if (i + 1) % show_nums == 0:
                logger.info(f"\tIter [{i+1}/{self.nums_epoch}] Losses: {losses_avg} Gradnorm: {gradnorm_avg}")
                
            if (i + 1) == self.nums_epoch:
                if self.args.lr_scheduler == 'TimmScheduler':
                    lrs = self.lrs._get_lr(self.epoch)[:2]
                else:
                    lrs = self.lrs.get_last_lr()[:2]
                if self.tb_writer is not None:
                    self.tb_writer.add_scalars('lrs', {'lr-'+str(i):lr for i,lr in enumerate(lrs)}, self.epoch+1)

                logger.info(f"\tIter [{i+1}/{self.nums_epoch}] Losses: {losses_avg} Gradnorm: {gradnorm_avg}")
                logger.info(f"Epoch [{self.epoch+1}/{self.args.epochs}] Lrs {[f'{lrs[i]:.4e}' for i in range(len(lrs))]} Losses: {losses_avg}")
                if self.tb_writer is not None:
                    self.tb_writer.add_scalars('losses', losses_avg.avgs, self.epoch+1)
                    
            self.iter_count += 1
            if self.iter_count <= self.args.lr_scheduler_kwargs['warmup_iters'] and self.args.lr_scheduler == 'LinearWarmupLrScheduler':
                if self.iter_count % show_nums == 0 or self.iter_count == self.args.lr_scheduler_kwargs['warmup_iters'] or self.iter_count == 1:
                    logger.info(f"\t\tIter [{self.iter_count}/{self.args.lr_scheduler_kwargs['warmup_iters']}] Warmup Lr: {self.lrs.get_last_lr()[0]}")
                self.lrs.step()
                
            # if self.iter_count == 2000 and self.args.amp and self.scaler.get_growth_interval() != 2000:
            #     logger.info(f"Setting AMP growth interval to 2000")
            #     self.scaler.set_growth_interval(2000)   
                
            if (i+1) == self.nums_epoch and self.args.sp_forever:
                break
 
        return True
    
    def train_test(self):
        best_mAP = 0.0
        start_epoch = self.epoch
        self.model.train()
        for self.epoch in range(start_epoch, self.args.epochs):
            # self.train_one_epoch(epoch, self.args.show_nums)
            if not self.train_one_epoch(self.args.show_nums):
                # print(self.model.pool.p)
                raise ValueError("NaN loss encountered")
            
            if self.args.lr_scheduler == 'TimmScheduler':
                self.lrs.step(self.epoch)
            else:
                if self.iter_count > self.args.lr_scheduler_kwargs['warmup_iters']:
                    self.lrs.step()

            if (self.epoch+1) in self.eval_epochs:
                metrics, metrics_flip = self.test(self.epoch)
                self.save_checkpoint(osp.join(self.savedir, f"ckpt_{self.epoch+1}.pth"))
                if metrics['mAP'] > best_mAP:
                    best_mAP = metrics['mAP']
                    
                if metrics_flip is not None and metrics_flip['mAP'] > best_mAP:
                    best_mAP = metrics_flip['mAP']
                    # torch.save(self.model.state_dict(), osp.join(self.savedir, f"best_model.pth"))
                
        if self.model_ema is not None:
            torch.save(self.model_ema.ema.state_dict(), osp.join(self.savedir, f"ema_model.pth"))
    
    def test(self, epoch=-1, ckpt=None):
        if ckpt is not None:
            if 'model' in ckpt:
                self.model.load_state_dict(ckpt['model'])
            else:
                self.model.load_state_dict(torch.load(ckpt))
        metrics = self.evaluator(epoch+1)
        self.model.train()
        return metrics
    
    def save_checkpoint(self, checkpoint_dir: str):
        ckpt_path = osp.join(checkpoint_dir, "ckpt.pth") if '.pth' not in checkpoint_dir else checkpoint_dir
        ckpt = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'lr_scheduler': self.lrs.state_dict(),
            'scaler': self.scaler.state_dict() if self.args.amp else None
        }
        torch.save(ckpt, ckpt_path)
        
    def load_checkpoint(self, checkpoint_dir: str):
        ckpt_path = osp.join(checkpoint_dir, "ckpt.pth") if '.pth' not in checkpoint_dir else checkpoint_dir
        ckpt = torch.load(ckpt_path)
        self.epoch = ckpt['epoch']
        self.model.load_state_dict(ckpt['model'])
        self.optim.load_state_dict(ckpt['optimizer'])
        self.lrs.load_state_dict(ckpt['lr_scheduler'])
        if self.args.args.amp and ckpt['scaler'] is not None:
            self.scaler.load_state_dict(ckpt['scaler'])
