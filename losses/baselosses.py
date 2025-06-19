import torch
from torch import nn
import torch.nn.functional as F
from .build import LOSS_REGISTRY
from einops import rearrange
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import TripletMarginLoss, CircleLoss
from pytorch_metric_learning.miners import BatchEasyHardMiner
import math
import numpy as np

__all__ = ['cross_entropy_loss', 'triplet_loss', 'circle_loss_softmax', 'circle_loss', 'reg_loss', 'ratr_intra_loss', 'ratr_inter_loss']


@LOSS_REGISTRY.register()
class ratr_intra_loss(nn.Module):
    def __init__(self, N, PK=[16, 4], tau=0.1):
        super().__init__()
        self.N = N
        D = PK[1] - 1
        self.D = D
        self.PK = PK
        self.tau = tau
        
        i = torch.arange(D).view(1, -1).expand(D, D)
        j = i.t()
        m = i < j
        
        self.register_buffer('i', i[m].view(-1))
        self.register_buffer('j', j[m].view(-1))
        
        targets = torch.arange(PK[0]).repeat(1,PK[1]).view(PK[1],PK[0]).transpose(0,1).contiguous().view(-1)
        pos_idxs = targets == targets.unsqueeze(1)
        pos_idxs[torch.arange(targets.size(0)), torch.arange(targets.size(0))] = 0
        self.register_buffer('pos_idxs', pos_idxs)

    def _kendall_tau(self, x, y):
        concordant_pairs = torch.tanh((x[:, self.i] - x[:, self.j])/self.tau) * torch.tanh((y[:, self.i] - y[:, self.j])/self.tau)
        total_pairs = (self.D * (self.D - 1)) / 2.0
        return concordant_pairs.sum(-1).mean() / total_pairs
        
    def forward(self, inputs, targets):
        
        pos_dists = []
        for ins_norm in inputs:
            tmp = torch.mm(ins_norm, ins_norm.t())
            # tmp = ins_norm.pow(2).sum(dim=1, keepdim=True) +\
            #     ins_norm.pow(2).sum(dim=1, keepdim=True).t() -\
            #     2 * torch.mm(ins_norm, ins_norm.t())
            # tmp = tmp.clamp(min=1e-12).sqrt()
            
            pos_dists.append(tmp[self.pos_idxs].view(targets.size(0), self.D))
            
        pos_loss = 0.0
        for i in range(self.N):
            for j in range(i+1, self.N):
                pos_loss += self._kendall_tau(pos_dists[i], pos_dists[j])
        pos_loss /= (self.N * (self.N-1) / 2.0)
        
        return pos_loss



@LOSS_REGISTRY.register()
class ratr_inter_loss(nn.Module):
    def __init__(self, N, PK=[16, 4], tau=0.1):
        super().__init__()
        self.N = N
        D = PK[0] - 1
        self.D = D
        self.PK = PK
        self.tau = tau
        
        i = torch.arange(D).view(1, -1).expand(D, D)
        j = i.t()
        m = i < j
        
        self.register_buffer('i', i[m].view(-1))
        self.register_buffer('j', j[m].view(-1))
        
        targets = torch.arange(PK[0]).repeat(1,PK[1]).view(PK[1],PK[0]).transpose(0,1).contiguous().view(-1)
        neg_idxs = targets.unsqueeze(1) != torch.arange(PK[0])
        self.register_buffer('neg_idxs', neg_idxs)
        
        
        
    def _kendall_tau(self, x, y):
        concordant_pairs = torch.tanh((x[:, self.i] - x[:, self.j])/self.tau) * torch.tanh((y[:, self.i] - y[:, self.j])/self.tau)
        total_pairs = (self.D * (self.D - 1)) / 2.0
        return concordant_pairs.sum(-1).mean() / total_pairs
        
    def forward(self, inputs, targets):

        neg_dists = []
        for ins_norm in inputs:
            centers = F.normalize(torch.mean(ins_norm.view(*self.PK, -1), dim=1)) # PK[0] x d
            tmp = torch.mm(ins_norm, centers.t()) # B x PK[0]
            
            # centers = torch.mean(ins_norm.view(*self.PK, -1), dim=1) # PK[0] x d
            # tmp = ins_norm.pow(2).sum(dim=1, keepdim=True) +\
            #     centers.pow(2).sum(dim=1, keepdim=True).t() -\
            #     2 * torch.mm(ins_norm, centers.t())
            # tmp = tmp.clamp(min=1e-12).sqrt()
            neg_dists.append(tmp[self.neg_idxs].view(targets.size(0), self.D))

        neg_loss = 0.0
        for i in range(self.N):
            for j in range(i+1, self.N):
                neg_loss += self._kendall_tau(neg_dists[i], neg_dists[j])
        neg_loss /= (self.N * (self.N-1) / 2.0)
        
        return neg_loss


@LOSS_REGISTRY.register()
class reg_loss(nn.Module):
    def __init__(self, s=1.0, k=0.5):
        super().__init__()
        
        self.s = s
        self.k = k

    def forward(self, inputs, targets):

        cls_feats, oth_feats, HW = inputs
        cls_feats = F.normalize(cls_feats, dim=-1)
        oth_feats = F.normalize(oth_feats, dim=-1)
        maps = torch.matmul(cls_feats, oth_feats.transpose(1, 2)) # B x n_cls x n_oth
        values, _ = torch.topk(torch.relu(maps), int(self.k * HW[0] * HW[1]), dim=-1)
        kth_largest = values[:, :, -1].unsqueeze(-1)
        # with torch.no_grad() and torch.cuda.amp.autocast(enabled=False):
        #     tmp = F.sigmoid(self.s * (torch.relu(maps) - kth_largest)).mean(0)
        #     res = []
        #     for i in range(tmp.size(0)):
        #         res.append(torch.histc(tmp[i].float(), bins=20, min=0.0, max=1.0))
        #     res = torch.stack(res, dim=0)
        #     res = res.mean(0)
        #     print(res)
        
        mms = F.sigmoid(self.s * (torch.relu(maps) - kth_largest)).sum(-1).mean()
        
        if torch.isnan(mms):
            print(1)

        return mms
# class reg_loss(nn.Module):
#     def __init__(self, k, n) -> None:
#         super().__init__()
        
#         self.k = k
#         self.n = n
        
#     def forward(self, input, *args, **kwargs):
#         '''
#         input: B x (N x d) 
#         '''
#         input_ = input.view(input.size(0), self.n, -1)
#         corrs  = torch.matmul(input_, input_.transpose(1,2)) # B x N x N
        
#         return torch.norm(corrs - torch.eye(self.n).to(input.device) * self.k / self.n, p=1, dim=(1,2)).mean()
        


        


# @LOSS_REGISTRY.register()
# class cross_entropy_loss(nn.CrossEntropyLoss):
#     def __init__(self, label_smoothing=0.0) :
#         '''
#         label_smoothing: float, default 0.0
#             If greater than 0, smooth the labels by adding a small value to them.
#             This can help to prevent overfitting.
            
#             y_label_smoothing = (1 - label_smoothing) * y_onehot + label_smoothing / num_classes
#         '''
#         super().__init__(label_smoothing=label_smoothing)
        
#     def forward(self, input, target):
#         return super().forward(input, target)
    
    
@LOSS_REGISTRY.register()
class cross_entropy_loss(nn.Module):
    def __init__(self, label_smoothing=0.0) :
        '''
        label_smoothing: float, default 0.0
            If greater than 0, smooth the labels by adding a small value to them.
            This can help to prevent overfitting.
            
            y_label_smoothing = (1 - label_smoothing) * y_onehot + label_smoothing / num_classes
        '''
        super().__init__()
        self.labal_smoothing = label_smoothing
        
    def forward(self, input, target):
        log_probs = F.log_softmax(input, dim=1)    
        with torch.no_grad():
            targets = torch.ones_like(log_probs)
            targets *= self.labal_smoothing / (input.size(1) - 1)
            targets.scatter_(1, target.data.unsqueeze(1), (1 - self.labal_smoothing))

        loss = (-targets * log_probs).sum(dim=1)
        if torch.isnan(loss.mean()):
            print(1)
        return loss.mean()
    
    

        

# @LOSS_REGISTRY.register()
# class triplet_loss(nn.Module):
#     def __init__(self, margin=0.3, squared=False, normalize_embeddings=False, pos_mining='hard', neg_mining='hard') -> None:
#         super().__init__()
#         self.miner = BatchEasyHardMiner(pos_mining, neg_mining)
#         self.loss = TripletMarginLoss(margin, smooth_loss=(margin==0.0), distance=LpDistance(normalize_embeddings=normalize_embeddings, p=2, power=1 if not squared else 2))
        
#     def forward(self, input, target):
#         hard_pairs = self.miner(input, target)
#         return self.loss(input, target, hard_pairs)

@LOSS_REGISTRY.register()
class circle_loss(nn.Module):
    def __init__(self, scale=128.0, margin=0.25):
        super().__init__()
        
        self.loss = CircleLoss(margin, scale)
        
    def forward(self, input, target):
        return self.loss(input, target)
        
@LOSS_REGISTRY.register()
class triplet_loss(nn.Module):
    def __init__(self, margin=0.3, squared=False, normalize_embeddings=False, pos_mining='hard', neg_mining='hard'):
        super().__init__()
        
        self.margin = margin
        self.squared = squared
        self.normalize_embeddings = normalize_embeddings
        self.pos_mining = pos_mining
        self.neg_mining = neg_mining        
        
    def forward(self, input, target):
        '''
            input: (N, D)
            target: (N) [0,0,0,0,1,1,1,1,2,2,2,2,...] PK-Sampling
        '''
        N, _ = input.size()
        
        if self.normalize_embeddings:
            input = F.normalize(input, p=2, dim=1)
        
        dists = input.pow(2).sum(dim=1, keepdim=True) +\
                input.pow(2).sum(dim=1, keepdim=True).t() -\
                2 * torch.mm(input, input.t())
        if not self.squared:
            dists = dists.clamp(min=1e-12).sqrt() # N x N
        
        pos_idxs = target == target.unsqueeze(1)
        # dists_ap = rearrange(dists[pos_idxs], '(N k) -> N k ', N = N).max(dim=1)[0]
        # dists_an = rearrange(dists[~pos_idxs], '(N k) -> N k ', N = N).min(dim=1)[0]
        if self.pos_mining == 'hard':
            dists_ap = dists[pos_idxs].view(N,-1).max(dim=1)[0]
        elif self.pos_mining == 'easy':
            toN = torch.arange(N).to(target.device)
            tmp = dists.clone().detach()
            tmp[toN, toN] = torch.inf
            idx = tmp[pos_idxs].view(N,-1).min(dim=1)[1]
            dists_ap = dists[pos_idxs].view(N,-1)
            dists_ap = dists_ap[toN, idx]
        else:
            raise ValueError('pos_mining should be either "hard" or "easy"')
        
        if self.neg_mining == 'hard':
            dists_an = dists[~pos_idxs].view(N,-1).min(dim=1)[0]
        elif self.neg_mining == 'easy':
            dists_an = dists[~pos_idxs].view(N,-1).max(dim=1)[0]
        elif self.neg_mining == 'semihard':
            toN = torch.arange(N).to(target.device)
            tmp = dists[~pos_idxs].view(N,-1) - dists_ap.unsqueeze(1).detach()
            tmp = tmp.detach()
            tmp[tmp<=0.0] = torch.inf
            idx = tmp.min(dim=1)[1]
            dists_an = dists[~pos_idxs].view(N,-1)[toN, idx]

            if torch.isinf(dists_an).any():
                dists_ap = dists_ap[~torch.isinf(dists_an)]
                dists_an = dists_an[~torch.isinf(dists_an)]
            
        else:
            raise ValueError('neg_mining should be either "hard", "easy" or "semihard"')
        
        if self.margin:
            loss = F.relu(dists_ap - dists_an + self.margin)
        else:
            loss = F.softplus(dists_ap - dists_an)
            
        if torch.isnan(loss.mean()):
            print(1)
        
        return loss.mean()
        

@LOSS_REGISTRY.register()
class circle_loss_softmax(cross_entropy_loss):
    def __init__(self, scale=256.0, margin=0.25, label_smoothing=0.0):
        super().__init__(label_smoothing)
        
        self.s = scale
        self.m = margin
        
    def forward(self, logits, targets):
        alpha_p = torch.clamp_min(-logits.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(logits.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m

        m_hot = F.one_hot(targets, num_classes=logits.size()[1]).float().to(logits.device)

        logits_p = alpha_p * (logits - delta_p)
        logits_n = alpha_n * (logits - delta_n)

        logits = logits_p * m_hot + logits_n * (1 - m_hot)
        return super().forward(logits * self.s, targets)